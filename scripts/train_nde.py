import argparse
import csv
import os
import pickle
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import optax
import tensorflow as tf
import tensorflow_probability as tfp
from chainconsumer import ChainConsumer
from haiku._src.nets.resnet import ResNet18
from sbi_lens.config import config_lsst_y_10
from sbi_lens.normflow.models import (
    AffineSigmoidCoupling,
    AffineCoupling,
    ConditionalRealNVP,
)
from tqdm import tqdm

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_access_sbi_lens", type=str, default=".")
parser.add_argument("--path_to_access_ssnpe_desc_project", type=str, default=".")
parser.add_argument("--total_steps", type=int, default=10_000)
parser.add_argument("--score_weight", type=float, default=0)
parser.add_argument("--exp_id", type=str, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_flow_layers", type=int, default=4)
parser.add_argument("--n_bijector_layers", type=int, default=2)
parser.add_argument("--activ_fun", type=str, default="silu")
parser.add_argument("--proposal", type=str, default="ps")
parser.add_argument("--sbi_method", type=str, default="npe")
parser.add_argument("--lr_schedule", type=str, default="exp_decay")
parser.add_argument("--nf", type=str, default="smooth")
parser.add_argument("--bacth_size", type=int, default=256)
args = parser.parse_args()

######## PARAMS ########
print("PARAMS---------------")
print("---------------------")

batch_size = args.bacth_size
tmp = list(range(0, 101_000, 5000))
tmp[0] = 1000
nb_simulations_allow = tmp[int(args.exp_id[4:])]

print("total_steps:", args.total_steps)
print("simulation budget:", nb_simulations_allow)
print("score_weight:", args.score_weight)
print("exp_id:", args.exp_id[4:])
print("seed:", args.seed)
print("n_flow_layers:", args.n_flow_layers)
print("n_bijector_layers:", args.n_bijector_layers)
print("activ_fun:", args.activ_fun)
print("lr_schedule:", args.lr_schedule)
print("proposal:", args.proposal)
print("sbi method:", args.sbi_method)
print("nf type:", args.nf)
print("batch size:", args.bacth_size)

print("---------------------")
print("---------------------")

PATH = "_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
    args.sbi_method,
    args.proposal,
    args.total_steps,
    args.score_weight,
    nb_simulations_allow,
    args.seed,
    args.n_flow_layers,
    args.n_bijector_layers,
    args.activ_fun,
    args.lr_schedule,
    args.nf,
    args.bacth_size
)

os.makedirs(
    f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/experiments/exp{PATH}/save_params"
)
os.makedirs(
    f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/experiments/exp{PATH}/fig"
)


######## CONFIG LSST Y 10 ########
print("... prepare config lsst year 10")
dim = 6

N = config_lsst_y_10.N
map_size = config_lsst_y_10.map_size
sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b = config_lsst_y_10.b
z0 = config_lsst_y_10.z0

truth = config_lsst_y_10.truth

params_name = config_lsst_y_10.params_name_latex

print("done ✓")

######## LOAD OBSERVATION AND REFERENCES POSTERIOR ########
print("... load observation and reference posterior")


# load reference posterior
sample_ff = jnp.load(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/posterior_full_field__256N_10ms_27gpa_0.26se.npy"
)

# load observed mass map
m_data = jnp.load(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/m_data__256N_10ms_27gpa_0.26se.npy"
)

print("done ✓")

######## DATASET ########
print("... load dataset")

if args.sbi_method == "npe":
    if args.proposal == "prior":
        dataset = np.load(
            f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/data/LOADED&COMPRESSED_year_10_with_noise_score_density.npz",
            allow_pickle=True,
        )["arr_0"]

    else:
        dataset = np.load(
            f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/data/LOADED&COMPRESSED_year_10_with_noise_score_density_proposal.npz",
            allow_pickle=True,
        )["arr_0"]
elif args.sbi_method == "nle":
    if args.proposal == "prior":
        dataset = np.load(
            f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/data/LOADED&COMPRESSED_year_10_with_noise_score_conditional.npz",
            allow_pickle=True,
        )["arr_0"]

    else:
        dataset = np.load(
            f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/data/LOADED&COMPRESSED_year_10_with_noise_score_conditional_proposal.npz",
            allow_pickle=True,
        )["arr_0"]


if args.sbi_method == "npe" and args.proposal == "ps":
    probs = jnp.load(
        f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/data/probs_ps_likelihood.npy"
    )
    prob_max = jnp.max(probs)

else:
    probs = jnp.zeros(len(dataset.item()["y"]))
    prob_max = 0

inds = jnp.unique(jnp.where(jnp.isnan(dataset.item()["score"]))[0])
dataset_y = jnp.delete(dataset.item()["y"], inds, axis=0)
dataset_score = jnp.delete(dataset.item()["score"], inds, axis=0)
dataset_theta = jnp.delete(dataset.item()["theta"], inds, axis=0)
probs = jnp.delete(probs, inds, axis=0)

print("done ✓")

######## COMPRESSOR ########
print("... prepare compressor")

compressor = hk.transform_with_state(lambda y: ResNet18(dim)(y, is_training=False))

a_file = open(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/params_compressor/opt_state_resnet_vmim.pkl",
    "rb",
)
opt_state_resnet = pickle.load(a_file)

a_file = open(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/params_compressor/params_nd_compressor_vmim.pkl",
    "rb",
)
parameters_compressor = pickle.load(a_file)

m_data_comressed, _ = compressor.apply(
    parameters_compressor, opt_state_resnet, None, m_data.reshape([1, N, N, nbins])
)

print("done ✓")

######## CREATE NDE ########
print("... build nde")

if args.activ_fun == "silu":
    activ_fun = jax.nn.silu
elif args.activ_fun == "sin":
    activ_fun = jnp.sin

if args.nf == "smooth":
    if args.sbi_method == "npe":

        key = jax.random.PRNGKey(0)

        omega_c = dist.TruncatedNormal(0.2664, 0.2, low=0).sample(key, (1000,))
        omega_b = dist.Normal(0.0492, 0.006).sample(key, (1000,))
        sigma_8 = dist.Normal(0.831, 0.14).sample(key, (1000,))
        h_0 = dist.Normal(0.6727, 0.063).sample(key, (1000,))
        n_s = dist.Normal(0.9645, 0.08).sample(key, (1000,))
        w_0 = dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3).sample(key, (1000,))

        theta = jnp.stack([omega_c, omega_b, sigma_8, h_0, n_s, w_0], axis=-1)

        scale = jnp.std(theta, axis=0) / 0.07
        shift = jnp.mean(theta / scale, axis=0) - 0.5

    elif args.sbi_method == "nle":
        # how these qantities are comute (with dataset form prior)
        # scale_y = jnp.std(dataset_y, axis=0) / 0.07
        # shift_y = jnp.mean(dataset_y / scale_y, axis=0) - 0.5

        scale = jnp.array(
            [1.2179337, 1.9040986, 2.070386, 1.9527259, 1.0068599, 0.38351834]
        )
        shift = jnp.array(
            [-0.51705444, -0.39985067, -0.53731424, -0.3843186, -0.5539374, -0.33893403]
        )

    bijector_layers = [128] * args.n_bijector_layers

    bijector = partial(
        AffineSigmoidCoupling,
        layers=bijector_layers,
        activation=activ_fun,
        n_components=16,
    )

    NF = partial(ConditionalRealNVP, n_layers=args.n_flow_layers, bijector_fn=bijector)

    class NDE(hk.Module):
        def __call__(self, y):
            nvp = NF(dim)(y)
            return tfd.TransformedDistribution(
                nvp, tfb.Chain([tfb.Scale(scale), tfb.Shift(shift)])
            )

elif args.nf == "affine":
    bijector_layers = [128] * args.n_bijector_layers

    bijector = partial(AffineCoupling, layers=bijector_layers, activation=activ_fun)

    NF = partial(ConditionalRealNVP, n_layers=args.n_flow_layers, bijector_fn=bijector)

    class NDE(hk.Module):
        def __call__(self, y):
            return NF(dim)(y)


if args.nf == "affine" and args.sbi_method == "npe" and args.score_weight > 0:
    raise ValueError("NDE has to be smooth")


if args.sbi_method == "npe":
    nf_log_prob = hk.without_apply_rng(
        hk.transform(lambda theta, y: NDE()(y).log_prob(theta).squeeze())
    )
    nf_get_posterior_sample = hk.transform(
        lambda y: NDE()(y).sample(len(sample_ff), seed=hk.next_rng_key())
    )
elif args.sbi_method == "nle":
    nf_log_prob = hk.without_apply_rng(
        hk.transform(lambda theta, y: NDE()(theta).log_prob(y).squeeze())
    )


def log_prob_fn(params, theta, y):
    return nf_log_prob.apply(params, theta, y)


print("done ✓")

######## LOSSES & UPDATE FUN ########
print("... prepare loss and update functions")


def loss_nll(params, mu, batch, weight, score, weight_score):
    weight = jnp.clip(1.0 / jnp.exp(weight - prob_max), 0, 500)

    lp, out = jax.vmap(
        jax.value_and_grad(
            lambda theta, x: log_prob_fn(
                params, theta.reshape([1, dim]), x.reshape([1, dim])
            ).squeeze()
        )
    )(mu, batch)

    return (
        -jnp.mean(lp * weight)
        + weight_score * jnp.sum((out - score) ** 2, axis=-1).mean()
    )


@jax.jit
def update(params, opt_state, mu, batch, w, score, weight_score):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_nll)(
        params, mu, batch, w, score, weight_score
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return loss, new_params, new_opt_state


print("done ✓")

######## TRAIN ########
print("... TRAINING")

params = nf_log_prob.init(
    jax.random.PRNGKey(args.seed), 0.5 * jnp.ones([1, dim]), 0.5 * jnp.ones([1, dim])
)

nb_steps = args.total_steps - args.total_steps * 0.2
# optimizer
if args.lr_schedule == "p_c_s":
    lr_scheduler = optax.piecewise_constant_schedule(
        init_value=0.001,
        boundaries_and_scales={
            int(nb_steps * 0.4): 0.2,
            int(nb_steps * 0.6): 0.3,
            int(nb_steps * 0.8): 0.4,
            int(nb_steps * 1): 0.5,
        },
    )

elif args.lr_schedule == "exp_decay":
    lr_scheduler = optax.exponential_decay(
        init_value=0.001,
        transition_steps=nb_steps // 50,
        decay_rate=0.9,
        end_value=1e-5,
    )

optimizer = optax.adam(lr_scheduler)
opt_state = optimizer.init(params)

batch_loss = []
lr_scheduler_store = []
pbar = tqdm(range(args.total_steps + 1))

for batch in pbar:
    inds = np.random.randint(0, nb_simulations_allow, batch_size)
    ex_theta = dataset_theta[inds]
    ex_y = dataset_y[inds]
    ex_score = dataset_score[inds]
    ex_weight = probs[inds]

    if not jnp.isnan(ex_y).any():
        l, params, opt_state = update(
            params, opt_state, ex_theta, ex_y, ex_weight, ex_score, args.score_weight
        )

        batch_loss.append(l)
        lr_scheduler_store.append(lr_scheduler(batch))
        pbar.set_description(f"loss {l:.3f}")

        if jnp.isnan(l):
            break

print("done ✓")

print("... save params, make plots and sample posterior")
# save params
with open(
    f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/experiments/exp{PATH}/save_params/params_flow.pkl",
    "wb",
) as fp:
    pickle.dump(params, fp)

# save plot loss
plt.figure()
plt.plot(batch_loss[10:])
plt.title("Batch Loss")
plt.savefig(
    f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/experiments/exp{PATH}/fig/loss"
)

# save plot loss
plt.figure()
plt.plot(lr_scheduler_store)
plt.title("lr schedule")
plt.savefig(
    f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/experiments/exp{PATH}/fig/lr_schedule"
)

if args.sbi_method == "npe":
    # save contour plot
    sample_nd = nf_get_posterior_sample.apply(
        params,
        rng=jax.random.PRNGKey(43),
        y=m_data_comressed * jnp.ones([len(sample_ff), dim]),
    )

    idx = jnp.where(jnp.isnan(sample_nd))[0]
    sample_nd = jnp.delete(sample_nd, idx, axis=0)

elif args.sbi_method == "nle":
    print("... run mcmc for posterior sampling")

    key = jax.random.PRNGKey(10)
    omega_c = dist.TruncatedNormal(0.2664, 0.2, low=0).sample(key, (1000,))
    omega_b = dist.Normal(0.0492, 0.006).sample(key, (1000,))
    sigma_8 = dist.Normal(0.831, 0.14).sample(key, (1000,))
    h_0 = dist.Normal(0.6727, 0.063).sample(key, (1000,))
    n_s = dist.Normal(0.9645, 0.08).sample(key, (1000,))
    w_0 = dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3).sample(key, (1000,))

    theta = jnp.stack([omega_c, omega_b, sigma_8, h_0, n_s, w_0], axis=-1)
    prior_mean = jnp.mean(theta, axis = 0)

    @jax.vmap
    def unnormalized_log_prob(theta):
        oc, ob, s8, h0, ns, w0 = theta

        prior = dist.TruncatedNormal(0.2664, 0.2, low=0).log_prob(oc)
        prior += dist.Normal(0.0492, 0.006).log_prob(ob)
        prior += dist.Normal(0.831, 0.14).log_prob(s8)
        prior += dist.Normal(0.6727, 0.063).log_prob(h0)
        prior += dist.Normal(0.9645, 0.08).log_prob(ns)
        prior += dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3).log_prob(w0)

        likelihood = log_prob_fn(
            params,
            theta.reshape([1, dim]),
            jnp.array(m_data_comressed).reshape([1, dim]),
        )

        return likelihood + prior

    # Initialize the HMC transition kernel.
    num_results = int(3e4)
    num_burnin_steps = int(5e2)
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_prob,
            num_leapfrog_steps=3,
            step_size=1e-2,
        ),
        num_adaptation_steps=int(num_burnin_steps * 0.8),
    )

    # Run the chain (with burn-in).
    @jax.jit
    def run_chain():
        # Run the chain (with burn-in).
        samples, is_accepted = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=jnp.array(prior_mean) * jnp.ones([12, 6]),
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            seed=jax.random.PRNGKey(0),
        )

        return samples, is_accepted

    samples_hmc, is_accepted_hmc = run_chain()
    sample_nd = samples_hmc[is_accepted_hmc]
    for i in range(6):
        plt.clf()
        plt.plot(sample_nd[..., i])
        plt.savefig(
            f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/experiments/exp{PATH}/fig/chain_{i}"
        )
    inds = np.random.randint(0, len(sample_nd), len(sample_ff))

    sample_nd = sample_nd[inds, ...]

plt.figure()
c = ChainConsumer()
c.add_chain(
    sample_ff,
    parameters=params_name,
    name="Full Field HMC",
    shade_alpha=0.5,
)
c.add_chain(
    sample_nd,
    parameters=params_name,
    name="Full Field SBI",
    shade_alpha=0.5,
)
fig = c.plotter.plot(
    figsize=1.2,
    truth=truth,
    extents=[
        [t - 5 * np.std(sample_ff[:, i]), t + 5 * np.std(sample_ff[:, i])]
        for i, t in enumerate(truth)
    ],
)

plt.savefig(
    f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/experiments/exp{PATH}/fig/contour_plot_step{batch}"
)
jnp.save(
    f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/experiments/exp{PATH}/posteriors_sample",
    sample_nd,
)

print("done ✓")

print("... save info experiment")

field_names = [
    "experiment_id",
    "sbi_method",
    "proposal",
    "activ_fun",
    "lr_schedule",
    "total_steps",
    "nb_simulations",
    "score_weight",
    "n_flow_layers",
    "n_bijector_layers",
    "seed",
    "nf type",
    "batch size"
]
dict = {
    "experiment_id": f"exp{PATH}",
    "sbi_method": args.sbi_method,
    "proposal": args.proposal,
    "activ_fun": args.activ_fun,
    "lr_schedule": args.lr_schedule,
    "total_steps": args.total_steps,
    "nb_simulations": nb_simulations_allow,
    "score_weight": args.score_weight,
    "n_flow_layers": args.n_flow_layers,
    "n_bijector_layers": args.n_bijector_layers,
    "seed": args.seed,
    "nf type": args.nf,
    "batch size": args.bacth_size
}

with open(
    f"{args.path_to_access_ssnpe_desc_project}/ssnpe_desc_project/results/store_experiments.csv",
    "a",
) as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
    dict_object.writerow(dict)

print("done ✓")
