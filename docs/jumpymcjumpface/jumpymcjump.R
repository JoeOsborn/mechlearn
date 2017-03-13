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
library(ggdendro)
library(RColorBrewer)
library(timeline)
library(ggrepel)

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
rm(temp)
names(jumpdata)[ncol(jumpdata)] <- "up.gravity_combined"
controlOrNot <- as.integer(!is.na(extract$up.control_reset))
jumpdata <- cbind(jumpdata,controlOrNot)
rm(controlOrNot)
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
aggregate(jumpdata,by=list(jump_kMeansFit$cluster),FUN=mean)

# append cluster assignment to each game
jump_kMeans_cluster <- jump_kMeansFit$cluster
table(jump_kMeans_cluster)
#jumpdata <- data.frame(jumpdata, jump_kMeans_cluster)
jump_kMeans_profile <- jump_kMeansFit$centers
xtable(table(extract$Developer,jump_kMeans_cluster))

#----------------------------------------------
# Archetypal analysis of games
#----------------------------------------------
#Archetypes from HAs
jump_aa <- stepArchetypes(data = jumpdata, k = 1:15, verbose = FALSE, nrep = 4)
screeplot(jump_aa)
jump_aa <- bestModel(jump_aa[[3]])
pcplot(jump_aa, jumpdata)

#Find archetype for each game
jump_aa_cluster<-max.col(coef(jump_aa))
#Append to dataset
shared <- data.frame(shared,shared_aa_cluster)
names(shared)[ncol(shared)] <- "Archetype"
#Archetype counts
table(extract$Publisher,jump_aa_cluster)
#Saving archetype information for plotting
jump_aa_profile<-parameters(jump_aa)

#----------------------------------------------
# Hierarchical clustering
#----------------------------------------------
jump_agnes <- agnes(x = jumpdata, metric = "euclidian", method = "ward",stand = TRUE)
jump_agnes_dd <- as.dendrogram(jump_agnes)
ggdendrogram(jump_agnes_dd,rotate = TRUE)

#----------------------------------------------
# Generate timeline
#----------------------------------------------
jumpcluster <- cbind(extract,jumpdata,jump_kMeansFit$cluster)
names(jumpcluster)[ncol(jumpcluster)] <- "Cluster"

NEStimeline <- data.frame(" ","All",as.Date("1985-01-01"),as.Date("1994-12-31"))
names(NEStimeline) <- c("Period","Group","Start","End")

minDates <- aggregate(extract$ReleaseDate.US, by = list(extract$Publisher), min)
maxDates <- aggregate(extract$ReleaseDate.US, by = list(extract$Publisher), max)
publisherDates <- cbind(minDates,maxDates$x)
publisherDates <- cbind(rep(" ",nrow(publisherDates)),publisherDates)
names(publisherDates) <- names(NEStimeline)
#NEStimeline <- rbind(NEStimeline,publisherDates)
jumpcluster_sorted <- jumpcluster[order(jumpcluster$ReleaseDate.US),]
#jumpcluster_sorted <- rbind(jumpcluster_sorted[seq(1,nrow(shared_sorted), 4),],shared_sorted[seq(2,nrow(shared_sorted), 4),],shared_sorted[seq(3,nrow(shared_sorted), 4),],shared_sorted[seq(4,nrow(shared_sorted), 4),])
timeline(df = NEStimeline, events = jumpcluster_sorted, label.col = "Period", group.col = "Group", start.col = "Start", end.col = "End", event.col = "ReleaseDate.US", event.label.col = "name", event.label.method = 2, event.line = FALSE,event.spots = 100, event.group.col = "Cluster",num.label.steps = 52,event.above = TRUE)

#----------------------------------------------
# Generate table with year and cluster members
#----------------------------------------------
table_yearsAndClusters <- table(jumpcluster$Year,jumpcluster$Cluster)
yearsAndClusters <- as.data.frame(table_yearsAndClusters)
names(yearsAndClusters) <- c("Year","Cluster","Frequency")
proptable_yearsAndClusters = prop.table(table_yearsAndClusters,margin = 1)
xtable(print(proptable_yearsAndClusters),digits = 2)

#----------------------------------------------
# Generate plot with jump clusters over years
#----------------------------------------------
ggplot(yearsAndClusters, aes(x = Year, y = Frequency, group = Cluster, colour = Cluster)) + 
  geom_line() + 
  geom_point()
