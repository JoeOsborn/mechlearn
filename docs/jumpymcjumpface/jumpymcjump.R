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
                                                                `up-control_reset` = col_number()), 
                  trim_ws = TRUE)
#View(extract)

extract <- data.frame(extract)

# Drop variables that are all 0's
extract$ground_reset = NULL
extract$ground_mult = NULL
extract$ground_gravity = NULL

# Get relevant variables for analysis
#jumpdata <- extract[,c("down_reset","down_mult","down_gravity","up.fixed_reset","up.fixed_mult","up.fixed_gravity","ground_reset","ground_gravity","minHoldDuration","maxHoldDuration")]
jumpdata <- extract[,c("down_reset","down_mult","down_gravity","up.fixed_reset","up.fixed_mult","up.fixed_gravity","minHoldDuration","maxHoldDuration")]
temp <- extract$up.control_gravity
temp[which(is.na(temp))] <- extract$up.fixed_gravity[which(is.na(temp))]
jumpdata <- cbind(jumpdata,temp)
names(jumpdata)[ncol(jumpdata)] <- "up.gravity_combined"
rownames(jumpdata) <- extract$name

#----------------------------------------------
# Visualizations for exploring the data - commented out, we don't need this right now
#----------------------------------------------
#Lattice plot of all features in summary statistics
splom(jumpdata,main="All game characters")

#----------------------------------------------
# Unrotated PCA jumpdata
#----------------------------------------------
#Principal component analysis

jump_pcaFit <- PCA(X = jumpdata,ncol(jumpdata),scale.unit = TRUE)
summary(jump_pcaFit)
plot(jump_pcaFit$eig[,2],type="b", main = "Screeplot", xlab = "Component", ylab="Variance explained")
xtable(jump_pcaFit$eig)

# Print the contributions of the first 3 dimensions from PCA
xtable(as.data.frame(jump_pcaFit$var$contrib))

#----------------------------------------------
# kMeans clustering
#----------------------------------------------
#K-means clustering
wss <- (nrow(jumpdata*sum(apply(jumpdata,2,var))))
for (i in 2:15){
  wss[i] <- sum(kmeans(jumpdata, centers=i)$withinss)
}
rm(i)
plot(1:15, wss, type="b", xlab="Number of kMeans Clusters", ylab="Within groups sum of squares")
rm(wss)

# K-Means Cluster Analysis
jump_kMeansFit <- kmeans(jumpdata, 3) # 3 cluster solution
#shared_kMedioidsFit <-  pamk(shared[10:ncol(shared)-1])
#clusplot(shared[10:ncol(shared)-1], shared_kMeansFit$cluster, color=TRUE, shade=TRUE, labels=1, lines=0, main="Shared, kMeans clusters")
fviz_cluster(jump_kMeansFit,data = jumpdata, geom = "text", repel = TRUE)

# get cluster means
aggregate(shared[10:ncol(shared)-1],by=list(shared_kMeansFit$cluster),FUN=mean)

# append cluster assignment to each game
shared_kMeans_cluster <- shared_kMeansFit$cluster
table(shared_kMeans_cluster)
shared <- data.frame(shared, shared_kMeans_cluster)
shared_kMeans_profile <- shared_kMeansFit$centers
xtable(table(shared$Developer,shared$shared_kMeans_cluster))
