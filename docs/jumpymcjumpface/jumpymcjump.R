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
extract <- read_delim("~/DevProjects/jupyter/mechlearn/docs/jumpymcjumpface/HAs.tsv", 
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
#TODO: Imputation should be removed, PCA should be split up into four groups
#3 different jump types + whatever's shared between all of them

pcaFit <- PCA(X = extractImputed[6:ncol(extractImputed)],ncol(extractImputed)-5,scale.unit = TRUE)
summary(pcaFit)
plot(pcaFit$eig[,2],type="b", main = "Screeplot, PCA", xlab = "Component", ylab="Variance explained")

#----------------------------------------------
# kMeans clustering
#----------------------------------------------
#K-means clustering
wss <- (nrow(extractNumerics[2:ncol(extractNumerics)])-1)*sum(apply(extractNumerics[2:ncol(extractNumerics)],2,var))
for (i in 2:15){
  wss[i] <- sum(kmeans(extractNumerics[2:ncol(extractNumerics)], centers=i)$withinss)
}
rm(i)
plot(1:15, wss, type="b", xlab="Number of kMeans Clusters", ylab="Within groups sum of squares")
rm(wss)
# K-Means Cluster Analysis
kMeansFit <- kmeans(extractNumerics[2:ncol(extractNumerics)], 4) # 4 cluster solution
clusplot(extractNumerics[2:ncol(extractNumerics)], kMeansFit$cluster, color=TRUE, shade=TRUE, labels=1, lines=0, main="kMeans clusters")

# get cluster means
aggregate(extractNumerics[2:ncol(extractNumerics)],by=list(kMeansFit$cluster),FUN=mean)

# append cluster assignment to each player
player_kMeans_cluster <- kMeansFit$cluster
table(player_kMeans_cluster)
extractNumerics <- data.frame(extractNumerics, player_kMeans_cluster)

kMeans_profile <- kMeansFit$centers

#----------------------------------------------
# Archetypal analysis with player scores aggregated across maps 
#----------------------------------------------
#Archetypes in player summary statistics
player_aa <- stepArchetypes(data = extractNumerics[2:(ncol(extractNumerics)-1)], k = 1:15, verbose = FALSE, nrep = 4)
screeplot(player_aa)
player_aa5 <- bestModel(player_aa[[5]])

pcplot(player_aa5, extractNumerics[2:8])
#barplot(player_aa5, extractNumerics[2:(ncol(extractNumerics)-1)])

#Find archetype for each player
player_aa5_cluster<-max.col(coef(player_aa5))
#Append to dataset
extractNumerics <- data.frame(extractNumerics,player_aa5_cluster)
names(extractNumerics)[ncol(extractNumerics)] <- "Archetype"
#Archetype counts
table(player_aa5_cluster)
#Saving archetype information for plotting
aa_profile<-parameters(player_aa5)

#----------------------------------------------
# Plotting clusters and archetypes
#----------------------------------------------
#plots K-means
plot(pcaFit$ind$coord[,1:2], type="n", xlim=(c(-8,4)), ylim=c(-2,3))
text(pcaFit$ind$coord[,1:2], col=kMeansFit$cluster, labels=kMeansFit$cluster)
arrows(0, 0, 7*pcaFit$var$coord[,1], 7*pcaFit$var$coord[,2], col = "chocolate", angle = 15, length = 0.025)
text(2*pcaFit$var$coord[,1], 2*pcaFit$var$coord[,2], labels=names(extractNumerics)[2:8])

#plots archetypes
plot(pcaFit$ind$coord[,1:2], type="n", xlim=(c(-8,4)), ylim=c(-2,3))
text(pcaFit$ind$coord[,1:2], col=player_aa5_cluster, labels=player_aa5_cluster)
arrows(0, 0, 7*pcaFit$var$coord[,1], 7*pcaFit$var$coord[,2], col = "chocolate", angle = 15, length = 0.025)
text(2*pcaFit$var$coord[,1], 2*pcaFit$var$coord[,2], labels=names(extractNumerics)[2:8])

#plots K-means centroids and archetype profiles together
#profiles for K-means and archetypes added to original data as supplementary points in order to map the results in a two-dimensional principal component space
extractNumerics_clusters_archetypes<-rbind(extractNumerics[2:8],aa_profile,kMeans_profile)
row.names(extractNumerics_clusters_archetypes)<-NULL

pcaFit <- PCA(X = extractNumerics_clusters_archetypes,7, ind.sup = 37:45, graph = FALSE)
plot(pcaFit$ind$coord[,1:2], pch=".", cex=6, xlim=(c(-8,5)), ylim=c(-3,3))
text(pcaFit$ind.sup$coord[,1:2], labels=c("A1","A2","A3","A4","A5","K1","K2","K3","K4"), col=c(rep("blue",5),rep("red",4)),cex=1)