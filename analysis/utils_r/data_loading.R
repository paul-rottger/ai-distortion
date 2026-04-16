# Load phase 2 data and create standard splits.
#
# Arguments:
#   annotations_path  - path to annotations.csv
#   phase1_path       - path to proposition_responses.csv
#   extra_mutate      - optional function(data) -> data, applied after base
#                       mutations and before splitting (use for script-specific
#                       columns like model_, input_condition_, ordinal factors)
#
# Returns a named list for use with list2env():
#   data_unedited, data_edited, data_preferred, get_split_data

load_phase2_splits <- function(annotations_path, phase1_path, extra_mutate = NULL) {
  data <- read_csv(annotations_path, show_col_types = FALSE)
  phase_1_preferences <- read_csv(phase1_path, show_col_types = FALSE)

  data <- data %>%
    mutate(
      rater_id       = as.factor(rater_id),
      writer_id      = as.factor(writer_id),
      proposition_id = as.factor(proposition_id)
    )

  if (!is.null(extra_mutate)) {
    data <- extra_mutate(data)
  }

  data_unedited <- data %>%
    filter(paragraph_type %in% c("writer", "model")) %>%
    mutate(
      paragraph_type_ = as.factor(paragraph_type),
      paragraph_type_ = relevel(paragraph_type_, ref = "writer")
    )

  data_edited <- data %>%
    group_by(writer_id, proposition_id) %>%
    filter(!(paragraph_type == "model" & any(paragraph_type == "edited"))) %>%
    mutate(
      paragraph_type  = if_else(paragraph_type == "edited", "model", paragraph_type),
      paragraph_type_ = as.factor(paragraph_type),
      paragraph_type_ = relevel(paragraph_type_, ref = "writer")
    ) %>%
    ungroup()

  preferred_exclusions <- phase_1_preferences %>%
    filter(writer_preference == "original") %>%
    mutate(
      writer_id      = as.factor(writer_id),
      proposition_id = as.factor(proposition_id)
    ) %>%
    distinct(writer_id, proposition_id)

  data_preferred <- data_edited %>%
    anti_join(preferred_exclusions, by = c("writer_id", "proposition_id"))

  splits <- list(
    unedited  = data_unedited,
    edited    = data_edited,
    preferred = data_preferred
  )

  list(
    data_unedited  = data_unedited,
    data_edited    = data_edited,
    data_preferred = data_preferred,
    get_split_data = function(data_split) splits[[data_split]]
  )
}
