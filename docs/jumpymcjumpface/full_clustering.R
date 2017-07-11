library(fpc)
#----------------------------------------------
# kMeans clustering
#----------------------------------------------
#K-means clustering
wss <- (nrow(full[10:ncol(full)])-1)*sum(apply(full[10:ncol(full)-1],2,var))
for (i in 2:15){
  wss[i] <- sum(kmeans(full[10:ncol(full)-1], centers=i)$withinss)
}
rm(i)
plot(1:15, wss, type="b", xlab="Full, number of kMeans Clusters", ylab="Within groups sum of squares")
rm(wss)
# K-Means Cluster Analysis
full_kMeansFit <- kmeans(full[10:ncol(full)-1], 3) # 4 cluster solution
full_kMedioidsFit <-  pamk(full[10:ncol(full)-1])
#clusplot(full[10:ncol(full)-1], full_kMeansFit$cluster, color=TRUE, shade=TRUE, labels = 1, lines=0, main="Full, kMeans clusters")
row.names(full) <- full$name
fviz_cluster(full_kMeansFit,data = full[10:ncol(full)-1], geom = "text", repel = TRUE)

# get cluster means
aggregate(full[10:ncol(full)-1],by=list(full_kMeansFit$cluster),FUN=mean)

# append cluster assignment to each player
full_kMeans_cluster <- full_kMeansFit$cluster
table(full_kMeans_cluster)
full <- data.frame(full, full_kMeans_cluster)
full_kMeans_profile <- full_kMeansFit$centers
xtable(table(full$Developer,full$full_kMeans_cluster))

#----------------------------------------------
# Archetypal analysis of games
#----------------------------------------------
#Archetypes from HAs
full_aa <- stepArchetypes(data = full[10:(ncol(full)-2)], k = 1:15, verbose = FALSE, nrep = 4)
screeplot(full_aa)
full_aa <- bestModel(full_aa[[8]])
pcplot(full_aa, full[10:ncol(full)-2])

#Find archetype for each game
full_aa_cluster<-max.col(coef(full_aa))
#Append to dataset
full <- data.frame(full,full_aa_cluster)
names(full)[ncol(full)] <- "Archetype"
#Archetype counts
table(full_aa_cluster)
#Saving archetype information for plotting
full_aa_profile<-parameters(full_aa)

#----------------------------------------------
# Plotting clusters and archetypes
#----------------------------------------------
#plots K-means
plot(full_pcaFit$ind$coord[,1:2], type="n")
text(full_pcaFit$ind$coord[,1:2], col=full_kMeansFit$cluster, labels=full_kMeansFit$cluster)
arrows(0, 0, 7*full_pcaFit$var$coord[,1], 7*full_pcaFit$var$coord[,2], col = "chocolate", angle = 15, length = 0.025)
text(2*full_pcaFit$var$coord[,1], 2*full_pcaFit$var$coord[,2], labels=names(full)[10:ncol(full)-3])

#plots archetypes
plot(full_pcaFit$ind$coord[,1:2], type="n")
text(full_pcaFit$ind$coord[,1:2], col=full_aa_cluster, labels=full_aa_cluster)
arrows(0, 0, 7*full_pcaFit$var$coord[,1], 7*full_pcaFit$var$coord[,2], col = "chocolate", angle = 15, length = 0.025)
text(2*full_pcaFit$var$coord[,1], 2*full_pcaFit$var$coord[,2], labels=names(full)[10:ncol(full)-3])

#plots K-means centroids and archetype profiles together
#profiles for K-means and archetypes added to original data as supplementary points in order to map the results in a two-dimensional principal component space
#full_clusters_archetypes<-rbind(full[10:ncol(full)-3],full_aa_profile,full_kMeans_profile)
#row.names(full_clusters_archetypes)<-NULL
#full_pcaFit <- PCA(X = extractNumerics_clusters_archetypes,7, ind.sup = 37:45, graph = FALSE)
#plot(pcaFit$ind$coord[,1:2], pch=".", cex=6, xlim=(c(-8,5)), ylim=c(-3,3))
#text(pcaFit$ind.sup$coord[,1:2], labels=c("A1","A2","A3","A4","A5","K1","K2","K3","K4"), col=c(rep("blue",5),rep("red",4)),cex=1)