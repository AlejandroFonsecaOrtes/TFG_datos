# source('~/dcm_cvband_irregular.R')

library(CVTuningCov)
# Y: array,n*m_max*p, X: time points, list of length n
cvthreshold <- function(Y, X, h, Xs, threshold.min = 0.1, threshold.max = 0.6) {
  
  nn = dim(Y)[1]
  pp = dim(Y)[2]
  #############################################
  n.1 = round(nn - nn/log(nn))
  n.2 = nn - n.1
  N1 = 20 # number of random splits
  subj_in = matrix(0, nrow = N1, ncol = n.1)
  for (j in 1:N1) {
    a = sample(1:nn, n.1, replace = FALSE, prob = NULL)
    a = sort(a)
    res = (1:nn)[is.na(pmatch(1:nn, a))]
    subj_in[j, ] = a
  }
  
  ##################################################################################################################
  ### calculate the estimated covariance matrices for each point in Xs, we only use the soft thresholding rule here.
  ##################################################################################################################
  ns = length(Xs)
  thresh1 = numeric(ns)
  
  for (shu in 1:ns) {
    print(paste0('evaluate:', shu))
    x = Xs[shu]
    thresholding = seq(threshold.min, threshold.max, by = 0.02)
    error.thresh.soft = numeric(length(thresholding))
    for (thresh in 1:length(thresholding)) {
      print(paste0('threshold cv:', thresh))
      s = thresholding[thresh]
      sum.NWs.soft = 0
      for (j in 1:N1) {
        print(paste0('threshold cv run:', j))
        a = subj_in[j,]
        res = (1:nn)[is.na(pmatch(1:nn, a))]
        sigma.NW.jihe1 = dcm_smoothCov(Y[a,,], tall[a], h, x)
        sigma.NW.jihe.soft1 = sign(sigma.NW.jihe1) * (abs(sigma.NW.jihe1) - s) *
          ((abs(sigma.NW.jihe1) - s) > 0)
        sigma.NW.jihe.res1 = dcm_smoothCov(Y[res,,], tall[res], h, x)
        sum.NWs.soft = sum.NWs.soft + F.norm2(sigma.NW.jihe.soft1 - sigma.NW.jihe.res1)
      }
      error.thresh.soft[thresh] = sum.NWs.soft/N1
    }
    s.soft = thresholding[which.min(error.thresh.soft)]
    thresh1[shu] = s.soft
  }
  return(list('threshold' = thresh1, error=error.thresh.soft))
}
