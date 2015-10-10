#' Bolasso model.
#'
#' This function performs a Bolasso logistic regression model.
#'
#' @param x the predictor matrix
#' @param y the response variable, a factor object with values of 0 and 1 
#' @param BM the number of bootstrapping, with the default value 100
#' @param kfold the K-fold cross validation, with the default value 10
#' @export
#' @import glmnet
#' @references
#' [1] Bach, F.R. (2008). Bolasso: model consistent lasso estimation through the bootstrap.  
#' Proceedings of the 25th international conference on Machine learning. ACM. pp. 33:40. 
#' @examples
#' library(datasets)
#' head(iris)
#' X <- as.matrix(subset(iris,iris$Species!="setosa")[,-5])
#' Y <- as.factor(ifelse(subset(iris,iris$Species!="setosa")[,5]=='versicolor',0,1))
#' # Fit a Bolasso logistic regression model.
#' # The BM parameter in the following example is set as small value to reduce  
#' # the running time, however the default value is proposed. 
#' Bolasso.fit <- Bolasso(x=X, y=Y, BM=5)
#' # Variables selected by the Bolasso model.
#' Bolasso.fit$var.selected
Bolasso=function(x, y, BM=100, kfold=10){
    varx <- colnames(x)
    rowx <- nrow(x)
    n <- length(y)
    res <- vector("list", BM)
    if (rowx!=n){
      stop("The number of rows in x is not equal to the length of y!")
                }
    for(i in 1:BM){
        repeat{ 
            s <- sample(n, replace=TRUE)
            if(length(table(y[s])) >= 2 & length(table(y[-s])) >= 2)
                break
              }
        BoostrapX <- as.matrix(x[s, ])
        colnames(BoostrapX) <- colnames(x)
        BoostrapY <- y[s]
        cvfit <- cv.glmnet(x=BoostrapX, y=BoostrapY, type.measure="deviance", nfolds=kfold, family="binomial")
        model.final <- cvfit$glmnet.fit
        nzero <- as.matrix(coef(model.final, s=cvfit$lambda.min))
        nzero <- names(nzero[nzero[,1]!=0,])
        res[[i]] <- nzero[which(nzero!="(Intercept)")]
        cat("Boostrap ", i, ":", "\n")
        cat(res[[i]], "\n")
                  }
    for(i in 1:BM){
        if (length(res[[i]])!=0){
        varx <- intersect(res[[i]], varx)
        cat("Boostrap ", i, ":", "\n")
        cat(varx, "\n")
                               }
                  }
    Myresult <- list(var.selected=varx)
    return(Myresult)
}
