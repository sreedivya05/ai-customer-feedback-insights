import streamlit as st
import pandas as pd
import json
import os
import re
import io
from typing import Dict, List, Optional

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Customer Feedback & Product Insights",
    page_icon="💬",
    layout="wide",
)

st.title("💬 AI Customer Feedback & Product Insights")
st.caption(
    "Upload a CSV, understand each record semantically, generate natural customer "
    "feedback, classify themes, prioritize issues, and surface product insights."
)

# ============================================================
# SESSION STATE
# ============================================================

if "generated_df" not in st.session_state:
    st.session_state.generated_df = None

# ============================================================
# CONSTANTS
# ============================================================

TEXT_COLUMN_HINTS = [
    "feedback", "customer_feedback", "comment", "comments", "review",
    "customer_review", "description", "details", "message", "customer_message",
    "text", "content", "request", "customer_request", "issue", "problem",
    "complaint", "summary", "reason", "notes", "issue_description"
]

CATEGORY_COLUMN_HINTS = [
    "category", "type", "feedback_type", "issue_type", "request_type",
    "theme", "topic", "area", "department", "module"
]

METADATA_EXACT = {
    "id", "ticket_id", "case_id", "record_id", "customer_id", "user_id",
    "date", "created_at", "updated_at", "timestamp", "status", "priority",
    "severity", "level", "category_id", "internal", "owner", "assigned_to",
    "agent_id", "row_id", "index"
}

# ============================================================
# COLUMN HELPERS
# ============================================================

def normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def is_metadata_column(column_name: str) -> bool:
    name = normalize_column_name(column_name)

    if name in METADATA_EXACT:
        return True

    metadata_suffixes = [
        "_id", "_date", "_timestamp", "_status", "_priority",
        "_severity", "_owner", "_assigned_to"
    ]

    return any(name.endswith(suffix) for suffix in metadata_suffixes)


def find_column(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    # Exact matches first
    for hint in hints:
        if hint in normalized:
            return normalized[hint]

    # Then partial matches
    for normalized_name, original in normalized.items():
        for hint in hints:
            if hint in normalized_name:
                return original

    return None


def find_text_columns(df: pd.DataFrame) -> List[str]:
    columns = []

    for col in df.columns:
        if is_metadata_column(col):
            continue

        name = normalize_column_name(col)

        if any(hint in name for hint in TEXT_COLUMN_HINTS):
            columns.append(col)

    # If no obvious text columns exist, use string columns
    if not columns:
        for col in df.columns:
            if is_metadata_column(col):
                continue

            if pd.api.types.is_string_dtype(df[col]):
                columns.append(col)

    return columns


def clean_value(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {
        "", "nan", "none", "null", "n/a", "na", "-", "unknown"
    }:
        return ""

    return value


# ============================================================
# CSV LOADING
# ============================================================

def load_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)

    # Normal CSV
    try:
        df = pd.read_csv(uploaded_file)

        if df.shape[1] >= 1:
            return df
    except Exception:
        pass

    # More tolerant parser
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(
            uploaded_file,
            encoding="utf-8-sig",
            sep=None,
            engine="python",
        )

        if df.shape[1] >= 1:
            return df
    except Exception:
        pass

    # Headerless fallback
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = [f"Column_{i + 1}" for i in range(df.shape[1])]

    return df


# ============================================================
# ROW UNDERSTANDING
# ============================================================

def collect_row_information(
    row: pd.Series,
    text_columns: List[str],
    category_column: Optional[str],
) -> Dict[str, str]:
    """
    Build a compact semantic representation of a CSV row.

    We intentionally preserve useful source information while
    removing IDs/status metadata that should not become customer speech.
    """

    info = {}

    if category_column:
        category_value = clean_value(row.get(category_column, ""))
        if category_value:
            info["Category"] = category_value

    # Put obvious text/description columns first.
    for column in text_columns:
        value = clean_value(row.get(column, ""))

        if value:
            info[str(column)] = value

    # Add other meaningful non-metadata columns.
    for column in row.index:
        if column in info:
            continue

        if is_metadata_column(column):
            continue

        value = clean_value(row[column])

        if value:
            info[str(column)] = value

    return info


def row_to_semantic_text(row_info: Dict[str, str]) -> str:
    parts = []

    for key, value in row_info.items():
        parts.append(f"{key}: {value}")

    return " | ".join(parts)


# ============================================================
# SEMANTIC CLASSIFICATION
# ============================================================

def detect_feedback_type(text: str) -> str:
    t = text.lower()

    # Payment
    if any(w in t for w in [
        "payment", "paid", "pay", "charged", "deducted", "refund",
        "transaction", "card", "upi", "billing", "money"
    ]):
        if any(w in t for w in [
            "failed", "fail", "declined", "deducted", "charged",
            "wrong", "refund", "error"
        ]):
            return "Payment Problem"
        return "Payment"

    # Authentication
    if any(w in t for w in [
        "login", "log in", "sign in", "password", "otp",
        "verification", "account access"
    ]):
        return "Authentication"

    # Bug
    if any(w in t for w in [
        "crash", "crashes", "freeze", "freezes", "frozen",
        "not working", "doesn't work", "does not work", "error",
        "bug", "broken", "fails", "failure", "unable to"
    ]):
        return "Bug"

    # Performance
    if any(w in t for w in [
        "slow", "loading", "lag", "latency", "takes too long",
        "performance", "delay"
    ]):
        return "Performance"

    # Delivery / order
    if any(w in t for w in [
        "delivery", "delivered", "order", "shipment",
        "shipping", "arrived", "late"
    ]):
        return "Delivery"

    # Feature request
    if any(w in t for w in [
        "would like", "i'd like", "want", "need", "add",
        "please add", "feature request", "feature", "option",
        "allow", "support apple pay", "support dark mode"
    ]):
        return "Feature Request"

    # Notifications
    if any(w in t for w in [
        "notification", "notifications", "alert", "reminder",
        "email", "push message"
    ]):
        return "Notifications"

    # Search
    if any(w in t for w in [
        "search", "filter", "find", "results"
    ]):
        return "Search"

    # Support
    if any(w in t for w in [
        "customer support", "customer service", "support team",
        "help desk", "agent"
    ]):
        return "Customer Support"

    # Usability
    if any(w in t for w in [
        "confusing", "difficult", "hard to use", "navigation",
        "interface", "ui", "ux", "navigate"
    ]):
        return "Usability"

    # Positive
    if any(w in t for w in [
        "love", "like", "great", "good", "excellent", "helpful",
        "easy", "fast", "smooth", "better", "improved"
    ]):
        return "Positive Feedback"

    # Suggestion
    if any(w in t for w in [
        "suggest", "recommend", "could", "maybe", "improve",
        "enhance"
    ]):
        return "Suggestion"

    return "General Feedback"


def priority_from_type(text: str, feedback_type: str) -> str:
    t = text.lower()

    high_words = [
        "money deducted", "charged", "payment failed", "payment failure",
        "fraud", "security", "cannot login", "can't login", "account locked",
        "crash", "crashes", "checkout", "unable to purchase", "data loss",
        "duplicate charge", "wrong charge"
    ]

    medium_words = [
        "slow", "delay", "confusing", "difficult", "not working",
        "delivery", "notification", "search", "support", "refund"
    ]

    if any(w in t for w in high_words):
        return "High"

    if feedback_type in {"Bug", "Payment Problem"}:
        return "High"

    if any(w in t for w in medium_words):
        return "Medium"

    if feedback_type in {
        "Feature Request", "Suggestion", "Performance",
        "Usability", "Customer Support"
    }:
        return "Medium"

    return "Low"


# ============================================================
# LOCAL NATURAL-LANGUAGE GENERATION
# ============================================================

def extract_value(row_info: Dict[str, str], keywords: List[str]) -> str:
    for key, value in row_info.items():
        normalized_key = normalize_column_name(key)
        if any(k in normalized_key for k in keywords):
            return value
    return ""


def local_generate_feedback(row_info: Dict[str, str]) -> str:
    """
    Deterministic fallback.

    This is intentionally semantic rather than simply joining
    CSV columns together.
    """

    text = row_to_semantic_text(row_info)
    lower = text.lower()

    category = extract_value(
        row_info,
        ["category", "type", "theme", "topic", "area", "module"]
    )

    description = extract_value(
        row_info,
        [
            "description", "feedback", "comment", "review",
            "message", "request", "issue", "problem", "summary",
            "content", "details"
        ]
    )

    # Prefer the meaningful description for wording.
    source = description or category or text

    # PAYMENT
    if "deduct" in lower and ("fail" in lower or "failed" in lower):
        return "My payment failed, but the amount was still deducted from my account."

    if "refund" in lower and any(w in lower for w in [
        "waiting", "pending", "not received", "missing", "delay"
    ]):
        return "I'm still waiting for my refund, and I'd like to know when the amount will be returned."

    if "payment" in lower and any(w in lower for w in [
        "failed", "fail", "declined", "error"
    ]):
        return "I tried to make a payment, but the transaction failed and I couldn't complete my purchase."

    # CHECKOUT / BUG
    if "checkout" in lower and "freeze" in lower:
        return "I can't complete my order because the checkout page freezes whenever I click the Pay button."

    if "checkout" in lower and any(w in lower for w in [
        "error", "fail", "not working", "broken"
    ]):
        return "I can't complete my order because something goes wrong during checkout."

    if "crash" in lower:
        return "The app keeps crashing when I try to use this feature, so I can't complete the task."

    if "freeze" in lower or "frozen" in lower:
        return "The screen freezes when I try to use this feature, so I can't continue."

    if any(w in lower for w in [
        "not working", "doesn't work", "does not work", "broken"
    ]):
        return "This feature isn't working as expected, so I'm unable to complete the task."

    # PERFORMANCE
    if any(w in lower for w in [
        "slow", "loading", "lag", "latency", "takes too long", "delay"
    ]):
        return "This part of the product takes too long to load, which makes it harder to complete my task."

    # DARK MODE
    if "dark mode" in lower:
        return "I'd really like a dark mode option because I often use the app at night."

    # APPLE PAY
    if "apple pay" in lower:
        return "I'd like to be able to use Apple Pay because it would make checkout more convenient."

    # FEATURE REQUEST
    if any(w in lower for w in [
        "feature request", "would like", "i'd like", "please add",
        "want", "add a", "add an", "add support"
    ]):
        if category:
            return f"I'd like the product to support {category.lower()} because it would make the experience more convenient."
        return "I'd like this capability to be added because it would make the product more convenient to use."

    # AUTHENTICATION
    if "password" in lower and any(w in lower for w in [
        "reset", "forgot", "not working", "unable"
    ]):
        return "I couldn't reset my password successfully, so I'm still unable to access my account."

    if any(w in lower for w in [
        "login", "log in", "sign in"
    ]) and any(w in lower for w in [
        "fail", "failed", "unable", "problem", "issue", "not working"
    ]):
        return "I'm having trouble signing into my account and can't access the product normally."

    # DELIVERY
    if "delivery" in lower and any(w in lower for w in [
        "delay", "late", "waiting"
    ]):
        return "My order has been delayed longer than expected, and I'd like an update on the delivery."

    if "delivery" in lower:
        return "I'd like clearer and more reliable information about my order's delivery status."

    # NOTIFICATIONS
    if "notification" in lower and any(w in lower for w in [
        "late", "delay", "delayed"
    ]):
        return "I'm receiving notifications too late, which makes it difficult to rely on them for important updates."

    # SEARCH
    if "search" in lower or "filter" in lower:
        return "Finding the right items is more difficult than it should be, and better search or filtering would help."

    # SUPPORT
    if "support" in lower or "customer service" in lower:
        if any(w in lower for w in ["wait", "delay", "slow", "late"]):
            return "I had to wait too long for a response from customer support, which made resolving my issue frustrating."
        return "I contacted customer support for help and would like the issue to be resolved more effectively."

    # USABILITY
    if any(w in lower for w in [
        "confusing", "difficult", "hard to use", "navigation", "navigate"
    ]):
        return "I found this part of the product confusing and wasn't sure what I was supposed to do next."

    # POSITIVE
    if any(w in lower for w in [
        "love", "great", "excellent", "helpful", "smooth",
        "easy to use", "works well", "good experience"
    ]):
        return "I really like this experience because it makes the product easier and more convenient to use."

    # SUGGESTION
    if any(w in lower for w in [
        "suggest", "recommend", "could improve", "enhance"
    ]):
        return "I think this part of the product could be improved to make the experience smoother."

    # Generic but still customer-like
    if source:
        return f"I'd like a better experience with {str(source).strip('.').lower()}."

    return "I think this part of the product could be improved to make the experience clearer and easier to use."


# ============================================================
# OPENAI GENERATION
# ============================================================

def get_openai_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")

    if key:
        return key

    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return None


def generate_feedback_with_ai(
    row_info: Dict[str, str],
    api_key: str,
    model: str
) -> Optional[str]:

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        row_json = json.dumps(
            row_info,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
You are an experienced product analyst writing customer feedback.

Convert the following CSV record into ONE realistic, natural customer
feedback statement.

CRITICAL RULES:
- Understand the meaning of the record before writing.
- Do not simply concatenate fields.
- Do not repeat column names.
- Do not mention CSV, row, column, dataset, data, or metadata.
- Ignore IDs, dates, internal status values, and internal priority labels
  unless they are genuinely necessary to understand the customer experience.
- Preserve the actual meaning. Do not invent facts.
- Write from the customer's perspective.
- Use first person when natural.
- For a bug, clearly state what went wrong and what the customer cannot do.
- For a feature request, clearly state what the customer wants and why it helps.
- For positive feedback, explain what the customer likes.
- For a complaint, explain the actual frustration.
- For payment issues, preserve whether the payment failed, was charged,
  refunded, or otherwise affected the customer.
- Keep the result to 1 or 2 concise sentences.
- Avoid generic filler such as "This affects my overall experience."

Examples:

Input:
{{
  "Category": "Checkout",
  "Description": "Checkout freezes after clicking Pay"
}}
Output:
I can't complete my order because the checkout page freezes whenever I click the Pay button.

Input:
{{
  "Category": "Payment",
  "Description": "Customer reports money deducted after failed payment"
}}
Output:
My payment failed, but the amount was still deducted from my account.

Input:
{{
  "Category": "Feature Request",
  "Description": "Customer wants dark mode"
}}
Output:
I'd really like a dark mode option because I often use the app at night.

Input:
{{
  "Category": "Positive",
  "Description": "Customer likes the new interface"
}}
Output:
I really like the new interface because it feels cleaner and is much easier to navigate.

Now convert this record:

{row_json}

Return ONLY the customer feedback statement.
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise, realistic customer feedback "
                        "for product analytics."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.7,
            max_tokens=120,
        )

        result = response.choices[0].message.content.strip()

        result = result.strip('"').strip("'")

        if result:
            return result

    except Exception:
        return None

    return None


# ============================================================
# FEEDBACK TYPE + PRIORITY
# ============================================================

def classify_feedback(feedback: str) -> str:
    return detect_feedback_type(feedback)


def classify_row(row_info: Dict[str, str], generated_feedback: str) -> str:
    combined = row_to_semantic_text(row_info)

    # Use source semantics first, generated feedback second.
    source_type = detect_feedback_type(combined)

    if source_type != "General Feedback":
        return source_type

    return classify_feedback(generated_feedback)


def classify_priority(
    row_info: Dict[str, str],
    generated_feedback: str,
    feedback_type: str
) -> str:

    combined = (
        row_to_semantic_text(row_info)
        + " "
        + generated_feedback
    )

    return priority_from_type(combined, feedback_type)


# ============================================================
# PRODUCT INSIGHTS
# ============================================================

def build_insights(result_df: pd.DataFrame) -> Dict[str, object]:
    insights = {}

    type_counts = (
        result_df["Feedback Type"]
        .value_counts()
        .to_dict()
    )

    priority_counts = (
        result_df["Priority"]
        .value_counts()
        .to_dict()
    )

    insights["type_counts"] = type_counts
    insights["priority_counts"] = priority_counts

    high = result_df[
        result_df["Priority"].str.lower() == "high"
    ]

    insights["high_priority_count"] = len(high)

    feature_count = len(
        result_df[
            result_df["Feedback Type"] == "Feature Request"
        ]
    )

    insights["feature_request_count"] = feature_count

    return insights


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    use_ai = st.checkbox(
        "Use AI generation",
        value=True,
        help="Uses OpenAI when OPENAI_API_KEY is configured. "
             "If unavailable, the local semantic generator is used.",
    )

    model = st.text_input(
        "OpenAI model",
        value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        disabled=not use_ai,
    )

    api_key_available = bool(get_openai_key())

    if use_ai:
        if api_key_available:
            st.success("OpenAI API key detected.")
        
    st.divider()

    st.write("### Supported feedback")

    st.write(
        """
        • Bugs  
        • Complaints  
        • Feature requests  
        • Suggestions  
        • Positive feedback  
        • Payments  
        • Performance  
        • Delivery  
        • Customer support  
        • Authentication  
        • Notifications  
        • Search  
        • Usability  
        • General product feedback
        """
    )

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload your CSV file",
    type=["csv"],
)

# ============================================================
# MAIN APP
# ============================================================

if uploaded_file is not None:

    try:
        df = load_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        st.success(
            f"CSV loaded successfully — {len(df)} rows and "
            f"{len(df.columns)} columns."
        )

        text_columns = find_text_columns(df)
        category_column = find_column(df, CATEGORY_COLUMN_HINTS)

        with st.expander("🔍 View uploaded CSV", expanded=False):
            st.dataframe(
                df,
                use_container_width=True,
            )

        st.write("### 🔎 Detected structure")

        col1, col2 = st.columns(2)

        with col1:
            if text_columns:
                st.write(
                    "**Text columns:** "
                    + ", ".join(map(str, text_columns))
                )
            else:
                st.write("**Text columns:** None detected")

        with col2:
            st.write(
                "**Category column:** "
                + (str(category_column) if category_column else "None")
            )

        st.divider()

        if st.button(
            "✨ Generate Natural Customer Feedback",
            type="primary",
            use_container_width=True,
        ):

            progress = st.progress(0)
            status = st.empty()

            feedback_results = []
            feedback_types = []
            priorities = []

            total_rows = len(df)

            for index, (_, row) in enumerate(df.iterrows()):

                row_info = collect_row_information(
                    row,
                    text_columns,
                    category_column,
                )

                if not row_info:
                    feedback = (
                        "I don't have enough information about this "
                        "customer experience to provide meaningful feedback."
                    )
                else:
                    feedback = None

                    if use_ai and get_openai_key():
                        status.text(
                            f"🤖 Analyzing row {index + 1} of {total_rows}..."
                        )

                        feedback = generate_feedback_with_ai(
                            row_info,
                            get_openai_key(),
                            model,
                        )

                    if not feedback:
                        status.text(
                            f"🧠 Generating semantic fallback for row "
                            f"{index + 1} of {total_rows}..."
                        )

                        feedback = local_generate_feedback(row_info)

                feedback_type = classify_row(
                    row_info,
                    feedback,
                )

                priority = classify_priority(
                    row_info,
                    feedback,
                    feedback_type,
                )

                feedback_results.append(feedback)
                feedback_types.append(feedback_type)
                priorities.append(priority)

                progress.progress(
                    (index + 1) / total_rows
                )

            progress.empty()
            status.empty()

            result_df = df.copy()

            result_df["Generated Customer Feedback"] = feedback_results
            result_df["Feedback Type"] = feedback_types
            result_df["Priority"] = priorities

            st.session_state.generated_df = result_df

            st.success(
                "Customer feedback and product insights generated successfully!"
            )

    except Exception as e:
        st.error(f"Could not process the CSV: {e}")


# ============================================================
# RESULTS
# ============================================================

if st.session_state.generated_df is not None:

    result_df = st.session_state.generated_df

    st.divider()

    st.subheader("📝 Generated Customer Feedback")

    st.dataframe(
        result_df,
        use_container_width=True,
        height=500,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("📊 Product Overview")

    total = len(result_df)

    high_priority = len(
        result_df[
            result_df["Priority"] == "High"
        ]
    )

    feature_requests = len(
        result_df[
            result_df["Feedback Type"] == "Feature Request"
        ]
    )

    positive_feedback = len(
        result_df[
            result_df["Feedback Type"] == "Positive Feedback"
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Feedback", total)

    with col2:
        st.metric("High Priority", high_priority)

    with col3:
        st.metric("Feature Requests", feature_requests)

    with col4:
        st.metric("Positive Feedback", positive_feedback)

    # --------------------------------------------------------
    # FEEDBACK TYPE CHART
    # --------------------------------------------------------

    st.subheader("📌 Feedback Themes")

    type_counts = (
        result_df["Feedback Type"]
        .value_counts()
    )

    st.bar_chart(type_counts)

    # --------------------------------------------------------
    # PRIORITY CHART
    # --------------------------------------------------------

    st.subheader("🚦 Priority Distribution")

    priority_counts = (
        result_df["Priority"]
        .value_counts()
    )

    st.bar_chart(priority_counts)

    # --------------------------------------------------------
    # TOP PAIN POINTS
    # --------------------------------------------------------

    st.subheader("🔴 Top Product Pain Points")

    pain_point_types = {
        "Bug",
        "Payment Problem",
        "Performance",
        "Authentication",
        "Delivery",
        "Notifications",
        "Search",
        "Usability",
        "Customer Support",
    }

    pain_points = result_df[
        result_df["Feedback Type"].isin(pain_point_types)
    ]

    if pain_points.empty:
        st.info("No major pain-point categories were detected.")

    else:
        pain_counts = (
            pain_points["Feedback Type"]
            .value_counts()
            .reset_index()
        )

        pain_counts.columns = [
            "Pain Point",
            "Count",
        ]

        st.dataframe(
            pain_counts,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # FEATURE REQUESTS
    # --------------------------------------------------------

    st.subheader("💡 Feature Requests")

    features = result_df[
        result_df["Feedback Type"] == "Feature Request"
    ]

    if features.empty:
        st.info("No feature requests were detected.")

    else:
        st.dataframe(
            features[
                [
                    "Generated Customer Feedback",
                    "Priority",
                ]
            ],
            use_container_width=True,
        )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader("🎯 Product Recommendations")

    if high_priority > 0:
        st.write(
            f"🔴 **Prioritize high-impact issues:** "
            f"{high_priority} high-priority feedback items were identified."
        )

    if feature_requests > 0:
        st.write(
            f"🟡 **Review feature demand:** "
            f"{feature_requests} feature requests were identified."
        )

    if positive_feedback > 0:
        st.write(
            f"🟢 **Preserve successful experiences:** "
            f"{positive_feedback} positive feedback items were identified."
        )

    if total > 0:
        top_theme = (
            result_df["Feedback Type"]
            .value_counts()
            .idxmax()
        )

        top_theme_count = (
            result_df["Feedback Type"]
            .value_counts()
            .max()
        )

        st.write(
            f"📈 **Focus on the dominant theme:** "
            f"'{top_theme}' appears most frequently with "
            f"{top_theme_count} feedback items."
        )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    st.subheader("💬 Generated Feedback Examples")

    example_count = min(10, len(result_df))

    for _, row in result_df.head(example_count).iterrows():

        st.markdown(
            f"**{row['Feedback Type']} · {row['Priority']} Priority**"
        )

        st.write(
            row["Generated Customer Feedback"]
        )

        st.divider()

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.subheader("⬇️ Export")

    csv_data = result_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Generated CSV",
        data=csv_data,
        file_name="generated_customer_feedback.csv",
        mime="text/csv",
        use_container_width=True,
    )