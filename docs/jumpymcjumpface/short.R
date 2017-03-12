#----------------------------------------------
# kMeans clustering
#----------------------------------------------
#K-means clustering
wss <- (nrow(short[10:ncol(short)])-1)*sum(apply(short[10:ncol(short)-1],2,var))
for (i in 2:15){
  wss[i] <- sum(kmeans(short[10:ncol(short)-1], centers=i)$withinss)
}
rm(i)
plot(1:15, wss, type="b", xlab="short, number of kMeans Clusters", ylab="Within groups sum of squares")
rm(wss)
# K-Means Cluster Analysis
short_kMeansFit <- kmeans(short[10:ncol(short)-1], 3) # 3 cluster solution
short_kMedioidsFit <-  pamk(short[10:ncol(full)-1])
#clusplot(short[10:ncol(short)-1], short_kMeansFit$cluster, color=TRUE, shade=TRUE, labels=1, lines=0, main="Short, kMeans clusters")
row.names(short) <- short$name
fviz_cluster(short_kMeansFit,data = short[10:ncol(short)-1], geom = "text")


# get cluster means
aggregate(short[10:ncol(short)-1],by=list(short_kMeansFit$cluster),FUN=mean)

# append cluster assignment to each player
short_kMeans_cluster <- short_kMeansFit$cluster
table(short_kMeans_cluster)
short <- data.frame(short, short_kMeans_cluster)
short_kMeans_profile <- short_kMeansFit$centers
xtable(table(short$Developer,short$short_kMeans_cluster))

#----------------------------------------------
# Archetypal analysis of games
#----------------------------------------------
#Archetypes from HAs
short_aa <- stepArchetypes(data = short[10:(ncol(short)-2)], k = 1:15, verbose = FALSE, nrep = 4)
screeplot(short_aa)
short_aa <- bestModel(short_aa[[6]])
pcplot(short_aa, short[10:ncol(short)-2])

#Find archetype for each game
short_aa_cluster<-max.col(coef(short_aa))
#Append to dataset
short <- data.frame(short,short_aa_cluster)
names(short)[ncol(short)] <- "Archetype"
#Archetype counts
table(short_aa_cluster)
#Saving archetype information for plotting
short_aa_profile<-parameters(short_aa)

#----------------------------------------------
# Plotting clusters and archetypes
#----------------------------------------------
#plots K-means
plot(short_pcaFit$ind$coord[,1:2], type="n")
text(short_pcaFit$ind$coord[,1:2], col=short_kMeansFit$cluster, labels=short_kMeansFit$cluster)
arrows(0, 0, 7*short_pcaFit$var$coord[,1], 7*short_pcaFit$var$coord[,2], col = "chocolate", angle = 15, length = 0.025)
text(2*short_pcaFit$var$coord[,1], 2*short_pcaFit$var$coord[,2], labels=names(short)[10:ncol(short)-3])

#plots archetypes
plot(short_pcaFit$ind$coord[,1:2], type="n")
text(short_pcaFit$ind$coord[,1:2], col=short_aa_cluster, labels=short_aa_cluster)
arrows(0, 0, 7*short_pcaFit$var$coord[,1], 7*short_pcaFit$var$coord[,2], col = "chocolate", angle = 15, length = 0.025)
text(2*short_pcaFit$var$coord[,1], 2*short_pcaFit$var$coord[,2], labels=names(short)[10:ncol(short)-3])

#plots K-means centroids and archetype profiles together
#profiles for K-means and archetypes added to original data as supplementary points in order to map the results in a two-dimensional principal component space
#short_clusters_archetypes<-rbind(short[10:ncol(short)-3],short_aa_profile,short_kMeans_profile)
#row.names(short_clusters_archetypes)<-NULL
#short_pcaFit <- PCA(X = extractNumerics_clusters_archetypes,7, ind.sup = 37:45, graph = FALSE)
#plot(pcaFit$ind$coord[,1:2], pch=".", cex=6, xlim=(c(-8,5)), ylim=c(-3,3))
#text(pcaFit$ind.sup$coord[,1:2], labels=c("A1","A2","A3","A4","A5","K1","K2","K3","K4"), col=c(rep("blue",5),rep("red",4)),cex=1)