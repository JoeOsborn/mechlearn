library(timeline)
library(ggrepel)

shared <- cbind(shared,shared_kMeansFit$cluster)
names(shared)[ncol(shared)] <- "Cluster"

NEStimeline <- data.frame(" ","All",as.Date("1985-01-01"),as.Date("1994-12-31"))
names(NEStimeline) <- c("Period","Group","Start","End")

minDates <- aggregate(shared$ReleaseDate.US, by = list(shared$Publisher), min)
maxDates <- aggregate(shared$ReleaseDate.US, by = list(shared$Publisher), max)
publisherDates <- cbind(minDates,maxDates$x)
publisherDates <- cbind(rep(" ",nrow(publisherDates)),publisherDates)
names(publisherDates) <- names(NEStimeline)
#NEStimeline <- rbind(NEStimeline,publisherDates)
shared_sorted <- shared[order(shared$ReleaseDate.US),]
shared_sorted <- rbind(shared_sorted[seq(1,nrow(shared_sorted), 4),],shared_sorted[seq(2,nrow(shared_sorted), 4),],shared_sorted[seq(3,nrow(shared_sorted), 4),],shared_sorted[seq(4,nrow(shared_sorted), 4),])


timeline(df = NEStimeline, events = shared_sorted, label.col = "Period", group.col = "Group", start.col = "Start", end.col = "End", event.col = "ReleaseDate.US", event.label.col = "name", event.label.method = 2, event.line = FALSE,event.spots = 100, event.group.col = "Cluster",num.label.steps = 52,event.above = TRUE)

