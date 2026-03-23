import jax
import jax.numpy as jnp

from .abeles import abeles

jax.config.update("jax_enable_x64", True)

def one_layer_refl(thick, sld, rough, sub_sld, sub_rough, qs):
    thicks = jnp.array([0, thick])
    roughs = jnp.array([sub_rough, rough])
    slds = jnp.array([sub_sld, sld])
    return abeles(thicks, roughs, slds, qs)

