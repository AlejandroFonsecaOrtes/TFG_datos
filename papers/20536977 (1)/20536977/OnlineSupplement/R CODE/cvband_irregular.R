library(matrixcalc)
library(foreach)
library(doParallel)
# local linear smoothing
K <- function(x) {ifelse(abs(x)<=1, 3*(1-x^2)/4, 0)} # kernel function

# weights
weight_ele <- function(u, x, h){
  s0 <- 0; s1 <-0; s2 <- 0
  n <- length(u)
  for(i in 1:n){
    tempt <- u[[i]]
    s0 <- s0 + sum(K((tempt-x)/h))
    s1 <- s1 + sum(K((tempt-x)/h)*(tempt-x))
    s2 <- s2 + sum(K((tempt-x)/h)*(tempt-x)^2)
  }
  return(c(s0,s1,s2))
}

smoothMean <- function(data, tall, h, u){
  n <- dim(data)[1]; p <- dim(data)[2]
  ss <- weight_ele(tall, u, h)
  temp <- rep(0, p)
  for(i in 1:n){
    tall2 <- tall[[i]]
    if(all(abs(tall2-u)>h)){
      next
    }else{
      tind <- which(abs(tall2-u)<=h)
      l <- length(tind)
      ttemp <- tall2[tind]
      for(j in 1:l){
        temp <- (ss[3]*K((ttemp[j] - u)/h)-ss[2]*K((ttemp[j] - u)/h)*(ttemp[j]-u)) * data[i,,tind[j]] + temp
      }
    }
  }
  temp1 <- ss[1]*ss[3]-ss[2]^2
  if(temp1==0){
    temp1<- temp1+1/n^2
  }
  return(temp/temp1)
}

smoothCov <- function(data, tall, h, u){
  n <- dim(data)[1]
  p <- dim(data)[2]
  ss <- weight_ele(tall, u, h)
  temp <- matrix(0, p, p)
  for(i in 1:n){
    data1 <- data[i,,]
    tall2 <- tall[[i]]
    if(all(abs(tall2-u)>h)){
      next
    }else{
      tind <- which(abs(tall2-u)<=h)
      l <- length(tind)
      ttemp <- tall2[tind]
      for(j in 1:l){
        temp <- (ss[3]*K((ttemp[j] - u)/h)-ss[2]*K((ttemp[j] - u)/h)*(ttemp[j]-u)) * tcrossprod(data1[,tind[j],drop=F]) + temp
      }
    }
  }
  temp1 <- ss[1]*ss[3]-ss[2]^2
  if(temp1==0){
    temp1<- temp1+1/n^2
  }
  smooth_mean <- matrix(smoothMean(data, tall, h, u), p, 1)
  return(temp/temp1 - tcrossprod(smooth_mean))
}

# tune the bandwidth given rho=o
# data: n*p*m
# hlist: candidate bandwidth parameters
# tall: time points; r: dimension of PC's
# leave-one-curve-out
cvband <- function(data,tall,hlist,r,rho=0, num_cores=10){
  n <- dim(data)[1]; p <- dim(data)[2]
  hlen <- length(hlist)
  error <- rep(0, hlen)
  cores <- num_cores
  cl <- makeCluster(cores)
  registerDoParallel(cl)
  result <- foreach(l = 1:n, .combine=rbind, .export = c('K', 'weight_ele', 'smoothMean', 'smoothCov')) %dopar% {
    datatest <- data[l,,]
    datatrain <- data[-l,,]
    talltrain <- tall[-l]
    ts <- tall[[l]]
    # for reduced computation, a subset of time points
    m <- min(length(ts), 10)
    t_ind <- sample(length(ts), m, replace = F)
    error <- rep(0, hlen)
    for(j in 1:hlen){
      h <- hlist[j]
      for(i in 1:m){
        evalt_ind <- t_ind[i]
        u<-ts[evalt_ind]
        S <- smoothCov(datatrain, talltrain, h=h, u)
        temp <- eigen(S)$vectors[,1:r,drop=F]
        error[j] <- error[j] + norm(datatest[,evalt_ind] - tcrossprod(temp) %*% matrix(datatest[,evalt_ind], p, 1),'2')^2
      }
    }
    return(error)
  }
  stopCluster(cl)
  count <- sapply(tall, length)
  count <- ifelse(count>10, 10, count)
  error <- colSums(result)/sum(count)
  return(list(cvh=hlist[which.min(error)], error=error, hlist = hlist, r = r))
}