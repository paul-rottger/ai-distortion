SCALE_ATTRIBUTES = [
    "writer_knowledge",
    "writer_importance",
    "writer_confidence",
    "writer_stance_polarity",
    "paragraph_formality",
    "paragraph_clarity",
    "paragraph_informativeness",
    "paragraph_originality",
    "paragraph_relevance",
    "writer_optimism",
    "writer_community",
    "writer_friendliness",
    "writer_openness",
    "paragraph_hope",
    "paragraph_excitement",
    "paragraph_fear",
    "paragraph_disgust",
    "paragraph_anger",
    "writer_affect_x",
    "writer_affect_y",
]

ORDINAL_VARS = [
    "writer_education",
    "writer_english_skills",
    "writer_income",
    "writer_age_binned",
    "writer_english_first",
]

NOMINAL_VARS = [
    "writer_politicalParty",
    "writer_politicalIdeology",
    "writer_gender",
    "writer_race",
]

CATEGORICAL_VARS = ORDINAL_VARS + NOMINAL_VARS

MODEL_TERMS = [
    "model_anthropic/claude-sonnet-4",
    "model_deepseek/deepseek-chat-v3-0324",
    "model_openai/chatgpt-4o-latest",
]

INPUT_CONDITION_TERMS = [
    "input_condition_stance-based",
    "input_condition_bullets-based",
    "input_condition_rewrite",
    "input_condition_improve",
]

VARIABLE_GROUPS = {
    "scale": {
        "attributes": SCALE_ATTRIBUTES,
        "metric_column": "ame",
        "summary_metric": "average_absolute_ame",
        "null_value": 0.0,
        "zero_label": "zero AME(s)",
        "direction_label": "AME directions",
    },
    "ordinal": {
        "attributes": ORDINAL_VARS,
        "metric_column": "odds_ratio",
        "summary_metric": "average_odds_ratio",
        "null_value": 1.0,
        "zero_label": "unit odds ratio(s)",
        "direction_label": "odds-ratio directions",
    },
}

CATEGORICAL_LEVELS = {
    "writer_age_binned": ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    "writer_english_first": ["No", "Yes"],
    "writer_english_skills": ["Basic", "Intermediate", "Advanced", "Expert"],
    "writer_education": [
        "GCSEs or equivalent",
        "A-levels or equivalent",
        "Vocational qualification",
        "Undergraduate degree",
        "Postgraduate degree (Master's)",
        "Doctorate (PhD)",
    ],
    "writer_income": [
        "Under £15,000",
        "£15,000-£24,999",
        "£25,000-£34,999",
        "£35,000-£49,999",
        "£50,000-£74,999",
        "£75,000-£99,999",
        "£100,000+",
    ],
    "writer_politicalIdeology": [
        "Very Left-Wing",
        "Moderately Left-Wing",
        "Centrist",
        "Moderately Right-Wing",
        "Very Right-Wing",
        "Other",
    ],
    "writer_politicalParty": [
        "Labour",
        "Conservative",
        "Liberal Democrats",
        "Greens",
        "Scottish National Party",
        "Did not vote",
        "Not eligible to vote",
        "Other",
    ],
    "writer_gender": ["Male", "Female", "Other"],
    "writer_race": ["White", "Black", "Asian", "Mixed", "Other"],
}
