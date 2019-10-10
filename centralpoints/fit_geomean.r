library(tidyverse)
library(plotly)
hypogeomean <- read_csv("C:/Users/lkape/.julia/dev/Hypatia/centralpoints/hypogeomean.csv")
hypogoemean.notuniform <- read_csv("C:/Users/lkape/.julia/dev/Hypatia/centralpoints/hypogeomean_notuniform.csv")
fit <- lm(w ~ log(n) + exp(alpha), data = hypogeomean)
summary(fit)
plot(resid(fit))

hypogeomean.level2 <- hypogeomean %>%
  filter((n > 2) & (n <= 5))
hypogeomean.level2 %>%
  plot_ly(x= ~log(n), y = ~(alpha), z = ~w, type="scatter3d", mode="markers")
fit.level2 <- lm(w ~ log(n) + exp(alpha), data = hypogeomean.level2)
summary(fit.level2)
plot(resid(fit.level2))

hypogeomean.level3 <- hypogeomean %>%
  filter((n > 5) & (n <= 20))
fit.level3 <- lm(w ~ log(n) + exp(alpha), data = hypogeomean.level3)
hypogeomean.level3 %>%
  plot_ly(x= ~log(n), y = ~exp(alpha), z = ~w, type="scatter3d", mode="markers")
summary(fit.level3)
plot(resid(fit.level3))

hypogeomean.level4 <- hypogeomean %>%
  filter((n > 20) & (n <= 100))
fit.level4 <- lm(w ~ log(n) + exp(alpha), data = hypogeomean.level4)
hypogeomean.level4 %>%
  plot_ly(x= ~log(n), y = ~exp(alpha), z = ~w, type="scatter3d", mode="markers")
summary(fit.level4)
plot(resid(fit.level4))

hypogeomean.level5 <- hypogeomean %>%
  filter(n > 100)
fit.level5 <- lm(w ~ log(n) + exp(alpha), data = hypogeomean.level5)
hypogeomean.level5 %>%
  plot_ly(x= ~log(n), y = ~exp(alpha), z = ~w, type="scatter3d", mode="markers")
summary(fit.level5)
plot(resid(fit.level5))
     
     



