import jax.numpy as jnp


def abeles(thicks: jnp.ndarray, roughs: jnp.ndarray, slds: jnp.ndarray, qs: jnp.ndarray) -> jnp.ndarray:
    # Get relative SLDs and convert to 1e-6 A^-2
    d_rho = (slds - slds[0]).astype(jnp.complex64) * 1e-6
    # Calculate the wavevector in each layer
    kn = jnp.sqrt(((qs[: None] * 0.5) ** 2) - (4 * jnp.pi * d_rho))
    a, b = kn[:, :-1], kn[:, 1:]

    # Fresnel reflection coefficient r with Névot-Croce roughness correction
    r = (a - b) / (a + b) * jnp.exp(-2 * a * b * roughs[:, None] ** 2)

    C00 = jnp.exp(1j * a * jnp.array([0, *thicks])).T
    C11 = 1 / C00
    C10 = r.T * C00
    C01 = r.T * C11

    # Calculate the total transfer matrix C for each layer and each q
    M00, M10, M01, M11 = C00[0], C10[0], C01[0], C11[0]
    for c00, c10, c01, c11 in zip(C00[1:], C10[1:], C01[1:], C11[1:]):

        a = M00 * c00 + M10 * c01
        b = M00 * c10 + M10 * c11
        c = M01 * c00 + M11 * c01
        d = M01 * c10 + M11 * c11
        M00, M10, M01, M11 = a, b, c, d

    # Calculate the reflectivity R from the total transfer matrix
    r = M10 / M00
    return (r * r.conj()).real


def abeles_1l(
    thick:
)