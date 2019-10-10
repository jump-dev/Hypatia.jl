library(tidyverse)
library(plotly)
hypogeomean <- read_csv("C:/Users/lkape/.julia/dev/Hypatia/centralpoints/hypogeomean.csv")
fit <- lm(w ~ log(n) + exp(alpha), data = hypogeomean)
plot(resid(fit))

hypogeomean %>%
  plot_ly(x= ~log(n), y = ~exp(alpha), z = ~w, type="scatter3d", mode="markers")

hypogeomean %>%
  plot_ly(x= ~log(n), y = ~alpha, z = ~w, type="scatter3d", mode="markers")

