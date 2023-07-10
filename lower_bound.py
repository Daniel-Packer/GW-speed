import jax.numpy as jnp
from ott.geometry import pointcloud

def pairwise_dists(point_cloud : pointcloud, squared = False):
    norms = jnp.sum(jnp.square(point_cloud.x), axis = 1)
    dists_sq = jnp.maximum(norms[:, None] + norms[None, :] - 2 * point_cloud.x @ point_cloud.x.T, 0)
    if squared:
        return dists_sq
    else:
        return jnp.sqrt(dists_sq)

def histogram_distances(point_cloud_x : pointcloud, point_cloud_y : pointcloud, p = 1):
    """
    Computes the distances needed to compute the second lower bound from
    [Memoli 2011] in a vectorized and jitable format. The general strategy here is taken
    from the scipy function `scipy.stats._cdf_distance`.
    """
    dists_x = pairwise_dists(point_cloud_x)
    dists_y = pairwise_dists(point_cloud_y)

    N_x, N_y = dists_x.shape[0], dists_y.shape[0]
    dists_x_sorted = jnp.sort(dists_x, axis = 1)
    dists_y_sorted = jnp.sort(dists_y, axis = 1)

    dists_x_wide = jnp.tile(dists_x_sorted[:, None, :], [1, N_y, 1])
    dists_y_wide = jnp.tile(dists_y_sorted[None, :, :], [N_x, 1, 1])

    all_values = jnp.concatenate([dists_x_wide, dists_y_wide], axis = 2)
    all_values_sorter = jnp.argsort(all_values, axis = 2)

    # These two methods should be equivalent. I'm not sure which is faster
    # all_values_sorted = jnp.sort(all_values)
    all_values_sorted = jnp.take_along_axis(all_values, all_values_sorter, axis = 2) # I'm guessing this one

    deltas = jnp.diff(all_values_sorted, axis = 2)

    dist_x_proto_pdf = jnp.concatenate([jnp.ones(dists_x_wide.shape), jnp.zeros(dists_y_wide.shape)], axis = 2)
    # dist_y_proto_pdf = jnp.concatenate([jnp.zeros(dists_x_wide.shape), jnp.ones(dists_y_wide.shape)], axis = 2)

    dist_x_pdf = jnp.take_along_axis(dist_x_proto_pdf, all_values_sorter, axis = 2)
    # dist_y_pdf = jnp.take_along_axis(dist_y_proto_pdf, all_values_sorter, axis = 2)
    dist_y_pdf = 1 - dist_x_pdf

    dist_x_cdf = jnp.cumsum(dist_x_pdf / N_x, axis = 2)[:, :, :-1]
    dist_y_cdf = jnp.cumsum(dist_y_pdf / N_y, axis = 2)[:, :, :-1]
    if p == 1:
        hist_dis_xy = jnp.sum(jnp.multiply(jnp.abs(dist_x_cdf - dist_y_cdf), deltas), axis = 2)
    elif p == 2:
        hist_dis_xy = jnp.sqrt(2) * jnp.sqrt(jnp.sum(jnp.multiply(jnp.square(dist_x_cdf - dist_y_cdf), deltas), axis = 2))
    else:
        hist_dis_xy = jnp.power(jnp.sum(jnp.multiply(jnp.power(jnp.abs(dist_x_cdf - dist_y_cdf), p),
                                       deltas), axis = 2), 1/p)

    return hist_dis_xy