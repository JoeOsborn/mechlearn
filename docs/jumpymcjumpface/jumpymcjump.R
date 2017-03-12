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
library(factoextra)
library(readr)
library(xtable)

#----------------------------------------------
# Load data
#----------------------------------------------
extract <- read_delim("~/DevProjects/jupyter/mechlearn/docs/jumpymcjumpface/HAs.csv", 
                  "\t", escape_double = FALSE, col_types = cols(`Date Number` = col_date(format = "%m/%d/%y"), 
                                                                `ReleaseDate-US` = col_date(format = "%m/%d/%y"), 
                                                                `up-control_gravity` = col_number(), 
                                                                `up-control_reset` = col_double()), 
                  trim_ws = TRUE)
View(extract)

extract <- data.frame(extract)

# Drop variables that are all 0's
extract$ground_reset = NULL
extract$ground_mult = NULL
extract$ground_gravity = NULL

# Identify three different types for PCA
extract <- cbind(extract,NA)
names(extract)[ncol(extract)] <- c("jump_type")
extract[which(!is.na(extract$up.control_reset)),]$jump_type = "Full"
extract[which(extract$minHoldDuration == extract$maxHoldDuration),]$jump_type = "Short"

full <- extract[which(extract$jump_type == "Full"),]

short <- extract[which(extract$jump_type == "Short"),]
short$up.control_reset = NULL
short$up.control_gravity = NULL
short$up.fixed_mult = NULL

shared <- extract[,!apply(is.na(extract), 2, any)]

#----------------------------------------------
# Visualizations for exploring the data
#----------------------------------------------
#Lattice plot of all features in summary statistics
splom(extract[5:ncol(extract)],main="All games")

splom(full[5:ncol(full)],main="Full jumps")
splom(short[,!apply(is.na(short), 2, any)],main="Short jumps")
splom(shared[5:ncol(shared)],main="Shared features")

#----------------------------------------------
# Unrotated PCA for all 3 kinds
#----------------------------------------------
#Principal component analysis
#3 different jump types, short, full, and one with variables shared between all jumps

full_pcaFit <- PCA(X = full[10:ncol(full)-1],ncol(full)-10,scale.unit = TRUE)
summary(full_pcaFit)
plot(full_pcaFit$eig[,2],type="b", main = "Full, screeplot", xlab = "Component", ylab="Variance explained")

short_pcaFit <- PCA(X = short[10:ncol(short)-1],ncol(short)-10,scale.unit = TRUE)
summary(short_pcaFit)
plot(short_pcaFit$eig[,2],type="b", main = "Short, screeplot", xlab = "Component", ylab="Variance explained")

shared_pcaFit <- PCA(X = shared[10:ncol(shared)-1],ncol(shared)-10,scale.unit = TRUE)
summary(shared_pcaFit)
plot(shared_pcaFit$eig[,2],type="b", main = "Shared, screeplot", xlab = "Component", ylab="Variance explained")
