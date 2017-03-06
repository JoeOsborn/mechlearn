#----------------------------------------------
# Load libraries
#----------------------------------------------
library(lattice)
library(cluster)
library(archetypes)
library(ggplot2)
library(reshape2)
library(stringr)
library(RColorBrewer)
library(missMDA)
library(FactoMineR)
library(readr)

#----------------------------------------------
# Load data
#----------------------------------------------
library(readr)
extract <- read_delim("~/DevProjects/jupyter/mechlearn/docs/jumpymcjumpface/newExtract3.tsv", 
                          "\t", escape_double = FALSE, col_types = cols(`ReleaseDate-US` = col_date(format = "%m/%d/%y"), 
                                                                        down_fixed_reset = col_number(), 
                                                                        down_gravity = col_number(), down_prev_mult = col_number(), 
                                                                        down_reset = col_number(), ground_fixed_reset = col_number(), 
                                                                        ground_gravity = col_number(), ground_prev_mult = col_number(), 
                                                                        ground_reset = col_number(), maxHoldDuration = col_number(), 
                                                                        minHoldDuration = col_number(), `up-control_fixed_reset` = col_number(), 
                                                                        `up-control_gravity` = col_number(), 
                                                                        `up-control_prev_mult` = col_number(), 
                                                                        `up-control_reset` = col_number(), 
                                                                        `up-fixed_fixed_reset` = col_number(), 
                                                                        `up-fixed_gravity` = col_number(), 
                                                                        `up-fixed_prev_mult` = col_number(), 
                                                                        `up-fixed_reset` = col_number()), 
                          trim_ws = TRUE)
View(extract)

extract <- data.frame(extract)

#Fix scaling errors in reset variables
extract[,"up.control_fixed_reset"] <- extract[,"up.control_fixed_reset"]/3600
extract[,"up.fixed_fixed_reset"] <- extract[,"up.fixed_fixed_reset"]/3600
extract[,"ground_fixed_reset"] <- extract[,"ground_fixed_reset"]/3600
extract[,"down_fixed_reset"] <- extract[,"down_fixed_reset"]/3600
extract[,"down_reset"] <- extract[,"down_reset"]/3600
extract[,"up.control_reset"] <- extract[,"up.control_reset"]/3600
extract[,"up.fixed_reset"] <- extract[,"up.fixed_reset"]/3600
extract[,"ground_reset"] <- extract[,"ground_reset"]/3600

# Clean extract for observations with missing values in down_fixed_reset
extract <- extract[which(!is.na(extract$down_fixed_reset)),]
View(extract)

# Replace missing values with imputed data (DO WE WANT THIS?)

# Get just the numeric variables
# Drop up_control_prev_mult which is only 0's or missing
extractNumerics <- extract[,5:7]
extractNumerics <- cbind(extractNumerics,extract[,9:ncol(extract)])
extractImputedPCA <- imputePCA(extractNumerics,ncp = 2)
extractImputed <- data.frame(extractImputedPCA$completeObs)
# Rebuild the data set
extractImputed <- cbind(extract[,1:4],extractImputed)

#----------------------------------------------
# Visualizations for exploring the data
#----------------------------------------------
#Lattice plot of all features in summary statistics
splom(extractImputed[5:ncol(extractImputed)],main="All features")

#----------------------------------------------
# Unrotated PCA
#----------------------------------------------
#Principal component analysis
pcaFit <- PCA(X = extractImputed[6:ncol(extractImputed)],ncol(extractImputed)-5,scale.unit = TRUE)
summary(pcaFit)
plot(pcaFit$eig[,2],type="b", main = "Screeplot, PCA", xlab = "Component", ylab="Variance explained")