evalFun <- function(estU, trueU){
  p <- dim(estU)[1]; r <- dim(estU)[2]
  estpro <- tcrossprod(estU, estU)
  truepro <- tcrossprod(trueU, trueU)
  sinangle <- svd(tcrossprod(estpro, diag(rep(1,p))-truepro))$d[1:r]
  error <- norm(sinangle,'2')
  
  ind <- which(diag(estpro)>1e-6)
  varintrue <- which(diag(truepro)>1e-6)
  varouttrue <- (1:p)[-varintrue]
  tp <- length(intersect(ind, varintrue))
  indout <- (1:p)[-ind]
  tn <- length(intersect(indout, varouttrue))
  tpr <- tp/(length(varintrue))
  tnr <- tn/(length(varouttrue))
  return(c(error, tpr, tnr))
}