import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
from chainconsumer import ChainConsumer
from sbi_lens.config import config_lsst_y_10
import haiku as hk
from functools import partial
from sbi_lens.normflow.models import ConditionalRealNVP, AffineSigmoidCoupling
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions
from haiku._src.nets.resnet import ResNet18
import pickle

from sbi_lens.metrics.c2st import c2st

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--total_steps", type=int, default=10_000)
parser.add_argument("--score_weight", type=float, default=0)
parser.add_argument("--exp_id", type=str, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_flow_layers", type=int, default=4)
parser.add_argument("--n_bijector_layers", type=int, default=2)
parser.add_argument("--activ_fun", type=int, default=0)
parser.add_argument("--lr_schedule", type=int, default=1)

args = parser.parse_args()


PATH = "compute_metric_{}_{}_{}_{}_{}_{}_{}".format(
    args.total_steps,
    args.score_weight,
    args.exp_id[4:],
    args.n_flow_layers,
    args.n_bijector_layers,
    args.activ_fun,
    args.lr_schedule,
)

os.makedirs("./{}".format(PATH))


######## LOAD REQUIRED DATA ########
print("######## LOAD REQUIRED DATA ########")

# !! load dataset from prior !!
dataset = np.load(
    "./LOADED&COMPRESSED_year_10_with_noise_score_density.npz", allow_pickle=True
)["arr_0"]

m_data = jnp.load("./data/m_data__256N_10ms_27gpa_0.26se.npy")

a_file = open("./data/params_compressor/opt_state_resnet_vmim.pkl", "rb")
opt_state_resnet = pickle.load(a_file)

a_file = open("./data/params_compressor/params_nd_compressor_vmim.pkl", "rb")
parameters_compressor = pickle.load(a_file)

metric_deep_ensemble = pd.read_table("./metric_deep_ensemble.csv", ",")

######## PARAMETERS ########
print("######## PARAMETERS ########")
n = 100_000  # number of sample to draw from mixture of NF
m = 10_000  # nb of sample to evaluate log prob for the weight of the mixture
seed_store = [2, 4, 5, 6, 7, 8, 9]

dim = 6
truth = config_lsst_y_10.truth
params_name = config_lsst_y_10.params_name_latex
N = config_lsst_y_10.N
nbins = config_lsst_y_10.nbins

tmp = list(range(0, 101_000, 5000))
tmp[0] = 1000
nb_simulations_allow = tmp[int(args.exp_id[4:])]

if args.activ_fun == 0:
    activ_fun = "silu"
    activ_fun_fn = jax.nn.silu
elif args.activ_fun == 1:
    activ_fun = "sin"
    activ_fun_fn = jnp.sin


if args.lr_schedule == 0:
    lr_schedule = "p_c_s"

elif args.lr_schedule == 1:
    lr_schedule = "exp_decay"


######## COMPRESS OBSERVED DATA ########
print("######## COMPRESS OBSERVED DATA ########")

compressor = hk.transform_with_state(lambda y: ResNet18(dim)(y, is_training=False))

m_data_comressed, _ = compressor.apply(
    parameters_compressor, opt_state_resnet, None, m_data.reshape([1, N, N, nbins])
)


######## CREATE NDE ########
print("######## NDE ########")

from sbi_lens.normflow.models import ConditionalRealNVP, AffineSigmoidCoupling
import numpyro.distributions as dist

key = jax.random.PRNGKey(0)

omega_c = dist.TruncatedNormal(0.2664, 0.2, low=0).sample(key, (1000,))
omega_b = dist.Normal(0.0492, 0.006).sample(key, (1000,))
sigma_8 = dist.Normal(0.831, 0.14).sample(key, (1000,))
h_0 = dist.Normal(0.6727, 0.063).sample(key, (1000,))
n_s = dist.Normal(0.9645, 0.08).sample(key, (1000,))
w_0 = dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3).sample(key, (1000,))

theta = jnp.stack([omega_c, omega_b, sigma_8, h_0, n_s, w_0], axis=-1)

scale_theta = jnp.std(theta, axis=0) / 0.07
shift_theta = jnp.mean(theta / scale_theta, axis=0) - 0.5

normalized_p = tfb.Chain([tfb.Scale(scale_theta), tfb.Shift(shift_theta)]).inverse(
    theta
)


bijector_layers = [128] * args.n_bijector_layers

bijector_npe = partial(
    AffineSigmoidCoupling,
    layers=bijector_layers,
    activation=activ_fun_fn,
    n_components=16,
)

NF_npe = partial(
    ConditionalRealNVP, n_layers=args.n_flow_layers, bijector_fn=bijector_npe
)


class SmoothNPE(hk.Module):
    def __call__(self, y):
        nvp = NF_npe(dim)(y)
        return tfd.TransformedDistribution(
            nvp, tfb.Chain([tfb.Scale(scale_theta), tfb.Shift(shift_theta)])
        )


log_prob_nvp_nd = hk.without_apply_rng(
    hk.transform(lambda theta, y: SmoothNPE()(y).log_prob(theta).squeeze())
)

log_prob_fn = lambda params, theta, y: log_prob_nvp_nd.apply(params, theta, y)


######## FUNCTION TO COMPUTE STATISTICS ########
print("######## FUNCTION TO COMPUTE STATISTICS ########")


@jax.jit
def compute_mean_log_likelihood(params, mu, batch):
    return jnp.mean(log_prob_fn(params, mu, batch))


def get_sample_from_DE(store_params, store_sample, theta, batch, n):
    likelihood_store = []

    for params in store_params:
        likelihood_store.append(compute_mean_log_likelihood(params, theta, batch))
    likelihood_store = jnp.array(likelihood_store)
    likelihood_store_med = jnp.median(likelihood_store)
    likelihood_store = jax.nn.softplus(likelihood_store - likelihood_store_med)
    likelihood_sum = jnp.sum(likelihood_store)
    weight = likelihood_store / likelihood_sum

    cat_samples = tfd.Categorical(probs=weight).sample(n, seed=jax.random.PRNGKey(0))
    unique, counts = np.unique(cat_samples, return_counts=True)
    cat_dict = dict(zip(unique, counts))

    for j, i in enumerate(cat_dict):
        if j == 0:
            weighted_samples = jnp.array(store_sample)[i][: cat_dict[i]]
        else:
            weighted_samples = jnp.concatenate(
                [weighted_samples, jnp.array(store_sample)[i][: cat_dict[i]]], axis=0
            )

    return weighted_samples, weight, cat_dict


@jax.jit
def get_logprob_from_DE(store_params, theta, x, cat):
    distribution_log_probs = [
        log_prob_fn(params, theta.reshape([-1, dim]), x.reshape([-1, dim]))
        for params in store_params
    ]
    cat_log_probs = jax.nn.log_softmax(cat.logits_parameter())
    final_log_probs = [
        cat_lp + d_lp for (cat_lp, d_lp) in zip(cat_log_probs, distribution_log_probs)
    ]
    concat_log_probs = jnp.stack(final_log_probs, 0)
    log_sum_exp = tfp.math.reduce_weighted_logsumexp(concat_log_probs, axis=[0])
    return log_sum_exp


# compute neg log prob
def nlpt(params, theta, y, cat):
    return -get_logprob_from_DE(params, theta, y, cat).mean()


######## GET REFERENCE POSTERIOR ########
print("######## GET PARAMS NF ########")

store_posteriors_sample = []
store_posteriors_params = []
for seed in seed_store:
    new_table = metric_deep_ensemble.loc[
        (metric_deep_ensemble["activ_fun"] == activ_fun)
        & (metric_deep_ensemble["lr_schedule"] == lr_schedule)
        & (metric_deep_ensemble["total_steps"] == args.total_steps)
        & (metric_deep_ensemble["score_weight"] == 0)
        & (metric_deep_ensemble["n_flow_layers"] == args.n_flow_layers)
        & (metric_deep_ensemble["n_bijector_layers"] == args.n_bijector_layers)
        & (metric_deep_ensemble["nb_simulations"] == 100_000)
        & (metric_deep_ensemble["seed"] == seed)
    ]

    exp_id = list(new_table["experiment_id"])[0]
    path1 = "./results/{}/posteriors_sample.npy".format(str(exp_id))
    path2 = "./results/{}/save_params/params_ode_flow.pkl".format(str(exp_id))
    posterior_sample = np.load(path1)
    posterior_params = np.load(path2, allow_pickle=True)
    store_posteriors_sample.append(posterior_sample)
    store_posteriors_params.append(posterior_params)

posterior_reference, _, _ = get_sample_from_DE(
    store_posteriors_params,
    store_posteriors_sample,
    dataset.item()["theta"][:m],
    dataset.item()["y"][:m],
    n,
)

######## GET PARAMS NF ########
print("######## GET PARAMS NF ########")

plt.figure()
c = ChainConsumer()

store_posteriors_sample = []
store_posteriors_params = []
for seed in seed_store:
    new_table = metric_deep_ensemble.loc[
        (metric_deep_ensemble["activ_fun"] == activ_fun)
        & (metric_deep_ensemble["lr_schedule"] == lr_schedule)
        & (metric_deep_ensemble["total_steps"] == args.total_steps)
        & (metric_deep_ensemble["score_weight"] == args.score_weight)
        & (metric_deep_ensemble["n_flow_layers"] == args.n_flow_layers)
        & (metric_deep_ensemble["n_bijector_layers"] == args.n_bijector_layers)
        & (metric_deep_ensemble["nb_simulations"] == nb_simulations_allow)
        & (metric_deep_ensemble["seed"] == seed)
    ]

    exp_id = list(new_table["experiment_id"])[0]
    path1 = "./results/{}/posteriors_sample.npy".format(str(exp_id))
    path2 = "./results/{}/save_params/params_ode_flow.pkl".format(str(exp_id))
    posterior_sample = np.load(path1)
    posterior_params = np.load(path2, allow_pickle=True)
    store_posteriors_sample.append(posterior_sample)
    store_posteriors_params.append(posterior_params)

    c.add_chain(
        posterior_sample,
        parameters=params_name,
        name="Approx posterior",
        shade_alpha=0.2,
    )

fig = c.plotter.plot(
    figsize=1.2,
    truth=truth,
)
plt.savefig("./{}/contour_plot.pdf".format(PATH), transparent=True)

posterior, weight, cat_dict = get_sample_from_DE(
    store_posteriors_params,
    store_posteriors_sample,
    dataset.item()["theta"][:m],
    dataset.item()["y"][:m],
    n,
)


######## PLOT ########
print("######## PLOT ########")
c = ChainConsumer()
c.add_chain(posterior, parameters=params_name, name="Approx", shade_alpha=0.2)
c.add_chain(
    posterior_reference,
    parameters=params_name,
    name="Ground truth",
    linewidth=1.5,
    color="#111111",
    shade_alpha=0,
    linestyle="--",
)
c.configure(legend_kwargs={"fontsize": 20}, tick_font_size=8, label_font_size=20)
fig = c.plotter.plot(
    figsize=1.2,
    truth=truth,
    extents=[
        [
            t - 5 * np.std(posterior_reference[:, i]),
            t + 5 * np.std(posterior_reference[:, i]),
        ]
        for i, t in enumerate(truth)
    ],
)

plt.savefig("./{}/compare_contour_plot.pdf".format(PATH), transparent=True)


######## COMPUTE METRICS ########
print("######## COMPUTE METRICS ########")

c2st_metric = c2st(posterior_reference, posterior, seed=0, n_folds=5)

cat = tfd.Categorical(probs=weight)
nlp = nlpt(
    store_posteriors_params,
    dataset.item()["theta"][:1000],
    dataset.item()["y"][:1000],
    cat,
)

######## SAVE RESULTS ########
print("######## SAVE RESULTS ########")

import csv

field_names = [
    "activ_fun",
    "lr_schedule",
    "total_steps",
    "nb_simulations",
    "score_weight",
    "n_flow_layers",
    "n_bijector_layers",
    "c2st",
    "nlp",
]
dict = {
    "activ_fun": activ_fun,
    "lr_schedule": lr_schedule,
    "total_steps": args.total_steps,
    "nb_simulations": nb_simulations_allow,
    "score_weight": args.score_weight,
    "n_flow_layers": args.n_flow_layers,
    "n_bijector_layers": args.n_bijector_layers,
    "c2st": c2st_metric,
    "nlp": nlp,
}

with open("./store_metrics_DE.csv", "a") as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
    dict_object.writerow(dict)
