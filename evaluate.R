library(pcalg)

true_binary_effect <- function(treatment, outcome, amat, cpt,
                               n = 10000, seed = 0) {
    set.seed(seed)
    # Create mutilated graph
    mutilated <- amat
    mutilated[, treatment] <- 0 # remove parents
    bng <- bnlearn::as.bn(as(mutilated != 0, "graphNEL"))

    # Generate interventional data for do(treatment = 0)
    cpt_0 <- cpt
    cpt_0[[treatment]] <- matrix(c(1, 0), ncol = 2)
    bnfit_0 <- bnlearn::custom.fit(bng, dist = cpt_0)
    data_0 <- bnlearn::rbn(bnfit_0, n = n)
    data_0 <- match(as.matrix(data_0), LETTERS) - 1
    data_0 <- matrix(data_0, ncol = ncol(amat))
    # E[outcome | do(treatment = 0)]
    mean_0 <- mean(data_0[, as.numeric(outcome)])

    # Generate interventional data for do(treatment = 1)
    cpt_1 <- cpt
    cpt_1[[treatment]] <- matrix(c(0, 1), ncol = 2)
    bnfit_1 <- bnlearn::custom.fit(bng, dist = cpt_1)
    data_1 <- bnlearn::rbn(bnfit_1, n = n)
    data_1 <- match(as.matrix(data_1), LETTERS) - 1
    data_1 <- matrix(data_1, ncol = ncol(amat))
    # E[outcome | do(treatment = 1)]
    mean_1 <- mean(data_1[, as.numeric(outcome)])

    # ATE = E[outcome | do(treatment = 1)] - E[outcome | do(treatment = 0)]
    effect <- mean_1 - mean_0
    return(effect)
}


get_adjustmet_set <- function(treatment, outcome, amat) {
    tryCatch(
        {
            return(pcalg::optAdjSet(as(amat, "graphNEL"), treatment, outcome))
        },
        error = function(e) {
            adj_set <- pcalg::adjustment(
                t(amat), "cpdag", treatment, outcome, "canonical"
            )
            if (length(adj_set) == 0) {
                return(NULL)
            } else {
                return(unlist(adj_set))
            }
        }
    )
}


is_unidentifiable <- function(treatment, outcome, amat) {
    treatment <- as.numeric(treatment)
    outcome <- as.numeric(outcome)
    # Check if treatment is a non-ancestor of outcome
    if (expm::expm(amat)[treatment, outcome] == 0) {
        return(0)
    }
    s <- get_adjustmet_set(treatment, outcome, amat)
    if (is.null(s)) {
        return(1)
    }
    return(0)
}


shd <- function(oracle, estimate) {
    return(pcalg::shd(as(oracle, "graphNEL"), as(estimate, "graphNEL")))
}
