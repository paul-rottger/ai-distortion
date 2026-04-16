rating_attributes <- c(
  # Writer opinion
  "writer_knowledge",
  "writer_importance",
  "writer_confidence",
  "writer_stance_polarity",
  # Paragraph quality
  "paragraph_formality",
  "paragraph_clarity",
  "paragraph_informativeness",
  "paragraph_originality",
  "paragraph_relevance",
  # Writer personality & values
  "writer_optimism",
  "writer_community",
  "writer_friendliness",
  "writer_openness",
  # Writer emotional state
  "paragraph_hope",
  "paragraph_excitement",
  "paragraph_fear",
  "paragraph_disgust",
  "paragraph_anger",
  "writer_affect_x",
  "writer_affect_y"
)

ordinal_vars <- c(
  "writer_education",
  "writer_english_skills",
  "writer_income",
  "writer_age_binned",
  "writer_english_first"
)

nominal_vars <- c(
  "writer_politicalParty",
  "writer_politicalIdeology",
  "writer_gender",
  "writer_race"
)

# Ordered factor levels for ordinal variables (used in mutate() calls)
ordinal_levels <- list(
  writer_age_binned     = c("18-29", "30-39", "40-49", "50-59", "60-69", "70+"),
  writer_english_first  = c("No", "Yes"),
  writer_english_skills = c("Basic", "Intermediate", "Advanced", "Expert"),
  writer_education      = c(
    "GCSEs or equivalent",
    "A-levels or equivalent",
    "Vocational qualification",
    "Undergraduate degree",
    "Postgraduate degree (Master's)",
    "Doctorate (PhD)",
    "Other"
  ),
  writer_income         = c(
    "Under £15,000",
    "£15,000-£24,999",
    "£25,000-£34,999",
    "£35,000-£49,999",
    "£50,000-£74,999",
    "£75,000-£99,999",
    "£100,000+"
  )
)
