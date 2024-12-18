# Imports
if (!requireNamespace("gprofiler2", quietly = TRUE)) {
    install.packages("gprofiler2", repos = "https://cloud.r-project.org/")
}
library(gprofiler2)
library(ggplot2)
library(xtable)

# Get list of SHAP genes
shap_genes <- read.csv(file='data/interim/20241115_shap/real_data/selected_genes.csv', row.names = 1)


# Remove the final "_at" if it exists
genes <- sapply(shap_genes$selected_genes, function(x) sub("_at$", "", x))
length(genes)


# Use gprofiler on genes list
alpha <- 0.05
gostres <- gost(query = genes,
                organism = "hsapiens", ordered_query = FALSE,
                multi_query = FALSE, significant = TRUE, exclude_iea = FALSE,
                measure_underrepresentation = FALSE, evcodes = FALSE,
                user_threshold = alpha, correction_method = "g_SCS",
                domain_scope = "annotated", custom_bg = NULL,
                numeric_ns = "", sources = NULL, as_short_link = FALSE, highlight = TRUE)



# Terms to highlight:
p_thres <- 1e-16
terms<-gostres$result[which(gostres$result$p_value<p_thres),]$term_id




# Get a table of the results:
results_table <- publish_gosttable(
  gostres,
  highlight_terms = terms,
  use_colors = TRUE,
  show_columns = c("source", "term_name", "term_size", "intersection_size"),
  ggplot = TRUE,
  filename = 'data/interim/20241115_shap/real_data/gosttable.pdf')


# Convert list columns to character strings
gostres_result <- gostres$result
gostres_result[] <- lapply(gostres_result, function(x) {
  if (is.list(x)) {
    sapply(x, toString)
  } else {
    x
  }
})

# Save the results as a CSV file
write.csv(gostres_result, file = 'data/interim/20241115_shap/real_data/gosttable.csv', row.names = FALSE)
# Save the results as a TeX file:
tex_table <- xtable(gostres_result)
print(tex_table, file = 'data/interim/20241115_shap/real_data/gosttable.tex')



# Create the plot
results_table <- publish_gosttable(
  gostres,
  highlight_terms = terms,
  use_colors = TRUE,
  show_columns = c("source", "term_name", "term_size", "intersection_size"),
  ggplot = TRUE,
  filename = NULL
)

# Save the plot with specified dimensions
ggsave(
  filename = 'data/interim/20241115_shap/real_data/gosttable.png',
  plot = results_table,
  width = 10,  # Adjust the width as needed
  height = 8   # Adjust the height as needed
)
