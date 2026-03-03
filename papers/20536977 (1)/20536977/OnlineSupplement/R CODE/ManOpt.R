library(matrixcalc)
library(Matrix)
library(expm)
library(CVTuningCov)
library(MASS)

# proximal gradient method on manifold optimization

getDelta <- function(B, rhoeta, U){
  r <- dim(U)[2]
  if(r==1){
    Delta <- diag(ifelse(abs(B[,1])>rhoeta, 1, 0))
    C <- crossprod(U,Delta) %*% U
  }
  else{
    Delta <- diag(ifelse(abs(B[,1])>rhoeta, 1, 0))
    C <- crossprod(U,Delta) %*% U
    for(i in 2:r){
      Delta <- diag(ifelse(abs(B[,i])>rhoeta, 1, 0))
      C <- as.matrix(bdiag(C, crossprod(U,Delta) %*% U))
    }
  }
  Delta <- (abs(B)>rhoeta)
  return(list(Delta=Delta, C=C))
}

getD <- function(B, U, rho, eta){
  # proximal mapping 
  # 1/2\|U+D-B\|_F^2 + eta*rho\|U+D\|_1
  p <- dim(U)[1]; r <- dim(U)[2]
  D <- matrix(0, p, r)
  for(i in 1:p){
    for(j in 1:r){
      if(B[i,j]>eta*rho){
        D[i,j] <- B[i,j]-eta*rho
      }
      else if(B[i,j]<(-eta*rho)){
        D[i,j] <- B[i,j]+eta*rho
      }
      else{
        D[i,j] <- 0
      }
    }
  }
  return(D-U)
}

# update the direction (manifold)
# regularized semi-smooth Newton method
# S: p*p, covariance matrix (local smoothing); A: r*r,Lagrangian multiplier matrix
# U: p*r, eigenvectors; rho: the sparsity parameter
# eta: step-size (proximal)
updateDirection <- function(S, A, U, rho, eta, tol, max_iter=100){
  # E(D(A)) = D(A)^t U + U^t D(A) = 0
  p <- dim(U)[1]
  r <- dim(U)[2]; rdim <- r*(r+1)/2
  B <- U + 2*eta*crossprod(S,U) + 2*eta*tcrossprod(U,A)
  D <- getD(B, U, rho=rho, eta=eta)
  E <- crossprod(D, U) + crossprod(U, D)
  vechE <- vech(E); vechA <- vech(A) # r(r+1)/2 dimensional vector
  oldobj <- sqrt(F.norm2(E))
  iter <- 0
  while(oldobj^2>tol&iter<=max_iter){
    mu <- 0.2*max(min(oldobj, 0.1), 1e-11)
    Z <- D + U
    nnZ <- sum(Z!=0)
    delta <- getDelta(B, rho*eta, U)
    C <- delta$C
    Delta <- delta$Delta

    if(nnZ > rdim){
      G <- 4*eta*pinvdupmat %*% C %*% dupmat # generalized Jacobian
      lu.G <- lu.decomposition(G + mu*diag(rep(1, rdim)))
      d <- -solve(lu.G$U, solve(lu.G$L, vechE))
      # d <- -solve(G + mu*diag(rep(1, rdim))) %*% vechE
    }else{
      Ustack <- matrix(0, nnZ, r^2)
      dim <- 0
      for(i in 1:r){
        row_idx <- which(Delta[,i]==1)
        if(length(row_idx)==0){
          next
        }
        Ustack[(dim+1):(dim+length(row_idx)), ((i-1)*r+1):(i*r)] <- U[row_idx,]
        dim <- dim + length(row_idx)
      }
      V <- Ustack %*% dupmat
      X <- 4*eta*tcrossprod(pinvdupmat, Ustack)
      lu.G <- lu.decomposition(diag(nnZ)+1/mu*V%*%X)
      d <- -(1/mu*vechE - 1/mu^2 * X%*%(solve(lu.G$U, solve(lu.G$L, V%*%vechE))))
    }
    
    vechA_new <- vechA + d  
    A_new <- matrix(dupmat %*% vechA_new, nrow = r, ncol = r)
    B_new <- U + 2*eta*crossprod(S,U) + 2*eta*tcrossprod(U,A_new)
    D_new <- getD(B_new, U, rho=rho, eta=eta)
    E_new <- crossprod(D_new, U) + crossprod(U, D_new)
    vechE_new <- vech(E_new)
    newobj <- sqrt(F.norm2(E_new))
    
    step <- 1
    while((newobj^2) >= (oldobj^2*(1-0.001*step)) && step > 0.001){
      # line search
      step <- step*0.5
      vechA_new <- vechA + step*d  
      A_new <- matrix(dupmat %*% vechA_new, nrow = r, ncol = r)
      B_new <- U + 2*eta*crossprod(S,U) + 2*eta*tcrossprod(U,A_new)
      D_new <- getD(B_new, U, rho=rho, eta=eta)
      E_new <- crossprod(D_new, U) + crossprod(U, D_new)
      vechE_new <- vech(E_new)
      newobj <- sqrt(F.norm2(E_new))
    }
    D <- D_new
    A <- A_new
    vechA <- vechA_new
    B <- B_new; vechE <- vechE_new
    oldobj <- newobj
    iter <- iter + 1
  }
  if(iter>max_iter){
    stop_flag <- 1
  }else{
    stop_flag <- 0
  }
  return(list(D=D, A=A, oldobj=oldobj, stop_flag=stop_flag))
}


# retraction: exponential mapping(default)
# alpha : step size
# D: direction
retraction <- function(U, D, alpha=1, type='exp'){
  r <- dim(U)[2]
  switch(type, exp={
    temp <- crossprod(U, D)
    temp1 <- D - U %*% temp
    Q <- qr.Q(qr(temp1))
    R <- qr.R(qr(temp1))
    temp3 <- rbind(temp, R)
    temp4 <- rbind(-t(R), matrix(0, r, r))
    temp5 <- cbind(temp3, temp4)
    temp6 <- as.matrix(expm(alpha*temp5))
    newU <- tcrossprod(crossprod(t(cbind(U, Q)), temp6), 
                       cbind(diag(rep(1,r)), matrix(0, r, r)))
  }, 
  polar={
    D <- alpha * D
    temp <- diag(rep(1,r)) + crossprod(D, D)
    newU <- crossprod(t(U+D), as.matrix(sqrtm(solve(temp))))
  },
  qr={
    qr.U <- qr(U+alpha*D)
    Q <- qr.Q(qr.U)
    R <- qr.R(qr.U)
    newU <- Q %*% diag(diag(sign(R)))
  },
  svd={
    eig <- eigen(crossprod(U+alpha*D))
    newU <- (U+alpha*D) %*% (eig$vectors%*%diag(1/sqrt(eig$values))%*%t(eig$vectors))
  })
  return(newU)
}

getobjValue <- function(S, U, rho){
  temp1 <- -sum(diag(crossprod(U, S) %*% U))
  temp2 <- rho*sum(abs(U))
  return(temp1 + sum(temp2))
}

# Armijo line search procedure
ManPG <- function(S, U, D, rho, delta=0.5, alpha=1, gamma=0.5,type='exp'){
  old <- getobjValue(S, U, rho=rho)
  newU <- retraction(U, D, alpha = alpha, type = type)
  new <- getobjValue(S, newU, rho=rho)
  while(new > (old - delta*alpha*F.norm2(D)) && alpha>1e-4){
    alpha <- gamma * alpha
    newU <- retraction(U, D, alpha = alpha, type = type)
    new <- getobjValue(S, newU, rho=rho)
  }
  return(list(newU=newU, alpha=alpha))
}

# rho:sparsity parameter; r: dimension of PC's
# tall: common-vector; irregular-list
evalManPG <- function(data, h, u, tall, rho, r, type='exp', max_iter=1000){
  n <- dim(data)[1]; p <- dim(data)[2]
  num_inexact <- 0
  in_flag <- 0
  S <- smoothCov(data, tall, h=h, u)
  eta <- 1 / (2*max(eigen(S)$values)) # step-size (proximal)
  U <- eigen(S)$vectors[,1:r,drop=F]
  
  tol <- 1e-8*p*r
  inner_tol <- max(1e-13, min(1e-11, 1e-3*tol*eta^2))
  # the initial Lagrangian multiplier matrix A
  A <- matrix(0, r, r)
  oldobj <- getobjValue(S, U, rho=rho)
  out <- updateDirection(S, A, U, rho=rho, eta=eta, tol=inner_tol)
  D <- out[[1]]
  # A <- out$A
  if(out$stop_flag == 1){
    # sub problem not exact
    in_flag <- in_flag + 1
  }
  updateU <- ManPG(S, U, D, rho = rho, delta = 0.5/eta, type = type)
  newU <- updateU$newU
  alpha <- updateU$alpha
  if(alpha<1e-4){
    num_inexact <- num_inexact+1
  }
  newobj <- getobjValue(S, newU, rho=rho)
  iter <- 0
  while(F.norm2(D)>=(tol*eta^2) & iter<=max_iter){
    if(alpha<1e-4 && num_inexact>10){
      inner_tol <- max(5e-16, min(1e-14, 1e-5*tol*eta^2))
    }else{
      inner_tol <- max(1e-13, min(1e-11, 1e-3*tol*eta^2))
    }
    out <- updateDirection(S, A, newU, rho=rho, eta=eta, tol=inner_tol)
    D <- out[[1]]
    # A <- out$A
    if(out$stop_flag == 1){
      in_flag <- in_flag + 1
    }
    updateU <- ManPG(S, newU, D, rho = rho, delta = 0.5/eta, type = type)
    newU <- updateU$newU
    alpha <- updateU$alpha
    if(alpha<1e-4){
      num_inexact <- num_inexact+1
    }
    newobj <- getobjValue(S, newU, rho=rho)
    iter <- iter + 1
  }
  
  if(iter>max_iter){
    flag_maxiter <- 1
  }else{
    flag_maxiter <- 0
  }
  if(iter>max_iter && sqrt(F.norm2(D))/eta > 0.1){
    print('unsuccessful!')
    flag_succ <- 0
  }else{
    flag_succ <- 1
  }
    return(list(U=newU, obj=newobj, smoothS=S, D=D, h=h, evalt=u, rho=rho, 
                ftol1=F.norm2(D), ftol2=abs(oldobj - newobj), flag_maxiter=flag_maxiter,
                in_flag=in_flag, flag_succ=flag_succ)) 
}
