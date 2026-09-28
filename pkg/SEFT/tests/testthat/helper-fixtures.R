tiny_seft_fixture <- function(dims = c(4L, 4L, 4L)) {
    set.seed(20260825)
    z <- array(rnorm(prod(dims)), dim = dims)
    atlas <- array(rep(seq_len(2L), each = prod(dims) / 2L), dim = dims)
    list(z = z, atlas = atlas, mask = atlas > 0L)
}
