library(data.table)
library(haven)
library(dplyr)
library(forcats)
library(stringr)
library(mice)
library(checkmate)
df <- fread("ZA7702_v2-1-0_open-ended.csv")

df <- read_dta("ZA7702_v2-1-0.dta")


labels <- data.frame(
  Column = names(df), # Spaltennamen
  Description = sapply(df, function(x) attr(x, "label")) # Labels extrahieren
)

fwrite(labels, "labels.csv")

# Recode variables

GLES2021_recoded <- df %>%
 # dplyr::filter(q2d == 1) %>% # restrict to eligible voters
  mutate(
    age = 2021 - as.numeric(d2a), #
    female = ifelse(d1 == 2, 1,
                    ifelse(d1 == 1, 0, NA)), #
    edu = ifelse(d8m == 1 | d8l == 1 | d8k == 1 | d8a == 1, 5,
                 ifelse(d7 == 4 | d7 == 5, 4,
                        ifelse(d7 == 3 | d7 == 6, 3,
                               ifelse(d7 == 2, 2,
                                      ifelse(d7 == 1 | d7 == 9, 1, NA))))),
    emp = ifelse(d9 == 1 | d9 == 2 | d9 == 8 | d9 == 11, 1,
                 ifelse(d9 == 7 | d9 == 10 | d9 == 12, 2,#
                        ifelse(d9 == 3 | d9 == 4 | d9 == 5 | d9 == 6 | d9 == 9, 3, NA))),
    hhincome = ifelse(d63 > 0, d63, NA), #
    hhincome = ifelse(hhincome <= 5, 1,
                      ifelse(hhincome >= 6 & hhincome <= 10, 2,
                             ifelse(hhincome >= 11, 3, NA))),
    east = ifelse(ostwest2 == 0, 1, ## merkw[rdig]
                  ifelse(ostwest2 == 1, 0, NA)),
    religious = ifelse(d41 > 0, d41, NA), #
    leftright = ifelse(q35e == 1 | q35e == 2, 1,
                       ifelse(q35e == 3 | q35e == 4, 2,
                              ifelse(q35e == 5 | q35e == 6 | q35e == 7, 3,
                                     ifelse(q35e == 8 | q35e == 9, 4,
                                            ifelse(q35e == 10 | q35e == 11, 5, NA))))), #
    partyid = ifelse(q75a == 1 | q75a == 2 | q75a == 3, 1, #
                     ifelse(q75a == 4, 2,
                            ifelse(q75a == 5, 3,
                                   ifelse(q75a == 6, 4,
                                          ifelse(q75a == 7, 5,
                                                 ifelse(q75a == 322, 6,
                                                        ifelse(q75a == 801, 7,
                                                               ifelse(q75a == 808, 8, NA)))))))),
    partyid_degree = ifelse(q76 == -97, 6, #
                            ifelse(q76 > 0, q76, NA)),
    inequality = ifelse(q102 == 1 | q102 == 2, 1, #
                        ifelse(q102 == 3, 2,
                               ifelse(q102 == 4 | q102 == 5, 3, NA))),
    immigration = ifelse(q43 > 0, q43, NA), #
    immigration = ifelse(immigration <= 5, 1,
                         ifelse(immigration == 6, 2,
                                ifelse(immigration >= 7, 3, NA))),
    vote = ifelse(q114ba == 1, 1,#
                  ifelse(q114ba == 4, 2,
                         ifelse(q114ba == 6, 3,
                                ifelse(q114ba == 5, 4,
                                       ifelse(q114ba == 7, 5,
                                              ifelse(q114ba == 322, 6,
                                                     ifelse(q114ba == 801, 7,
                                                            ifelse(q114ba == -83, 8,
                                                                   ifelse(q114ba == 2, 9, NA))))))))),
    female = factor(female, levels = c(0, 1), labels = c("männlich", 
                                                         "weiblich")),
    edu = factor(edu, levels = c(1, 2, 3, 4, 5), 
                 labels = c("keinen Schulabschluss",
                            "einen Hauptschulabschluss",
                            "einen Realschulabschluss",
                            "Abitur",
                            "einen Hochschulabschluss")),
    emp = factor(emp, levels = c(1, 2, 3), labels = c("berufstätig",
                                                      "nicht berufstätig",
                                                      "in Ausbildung")),
    hhincome = factor(hhincome, levels = c(1, 2, 3), 
                      labels = c("niedriges", "mittleres", "hohes")),
    east = factor(east, levels = c(0, 1), labels = c("Westdeutschland",
                                                     "Ostdeutschland")),
    religious = factor(religious, levels = c(1, 2, 3, 4), 
                       labels = c("überhaupt nicht religiös",
                                  "nicht sehr religiös",
                                  "etwas religiös",
                                  "sehr religiös")),
    leftright = factor(leftright, levels = c(1, 2, 3, 4, 5),
                       labels = c("stark links",
                                  "mittig links",
                                  "in der Mitte",
                                  "mittig rechts",
                                  "stark rechts")),
    partyid = factor(partyid, levels = c(1, 2, 3, 4, 5, 6, 7, 8),
                     labels = c("mit der Partei CDU/CSU",
                                "mit der Partei SPD", 
                                "mit der Partei FDP",
                                "mit der Partei Bündnis 90/Die Grünen",
                                "mit der Partei Die Linke",
                                "mit der Partei AfD",
                                "mit einer Kleinpartei",
                                "mit keiner Partei")),
    partyid_degree = factor(partyid_degree, levels = c(1, 2, 3, 4, 5, 6),
                            labels = c("sehr stark ",
                                       "ziemlich stark ",
                                       "mäßig ",
                                       "ziemlich schwach ",
                                       "sehr schwach ",
                                       "")),
    party = paste0(partyid_degree, partyid),
    inequality = factor(inequality, levels = c(1, 2, 3),
                        labels = c("Maßnahmen ergreifen",
                                   "habe keine Meinung dazu, ob die Regierung Maßnahmen ergreifen sollte",
                                   "keine Maßnahmen ergreifen")),
    immigration = factor(immigration, levels = c(1, 2, 3), 
                         labels = c("erleichtern",
                                    "weder erleichtern noch einschränken",
                                    "einschränken")),
    vote = factor(vote, levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9),
                  labels = c("CDU/CSU",
                             "SPD",
                             "Bündnis 90/Die Grünen",
                             "FDP",
                             "Die Linke",
                             "AfD",
                             "Andere Partei",
                             "Ungültig gewählt",
                             "Nicht gewählt"))
  ) %>%
  select(
    lfdn,
    age,
    female,
    edu,
    emp,
    hhincome,
    east,
    religious,
    leftright,
    partyid,
    partyid_degree,
    party,
    inequality,
    immigration,
    vote
  )