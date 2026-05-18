library(writexl)

# Import the object to change
obj <- readRDS("biographies_features_only.Rds")

# Create a function that will flatten the nested tables
flatten_tables <- function(x, prefix = "") {
  out <- list()
  
  for (name in names(x)) {
    new_name <- if (prefix == "") name else paste(prefix, name, sep = "_")
    item <- x[[name]]
    
    if (is.data.frame(item)) {
      clean_name <- gsub("[:\\\\/?*\\[\\]]", "_", new_name)
      clean_name <- substr(clean_name, 1, 31)
      
      out[[clean_name]] <- item
      
    } else if (is.list(item)) {
      out <- c(out, flatten_tables(item, new_name))
    }
  }
  
  return(out)
}

df_list <- flatten_tables(obj)

names(df_list)
length(df_list)

write_xlsx(df_list, "biographies_features_only.xlsx")

#write_xlsx(obj, "output.xlsx")