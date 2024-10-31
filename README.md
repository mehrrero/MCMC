# MCJax

This package contains a simple implementation of a Markov chain Monte Carlo algorithm in python using JAX. For the moment it only implements the Metropolis-Hasting algorithm, but allows for custom target distributions in any number of dimensions. Thanks to JAX, the code runs in parallel and it is both CPU and GPU ready, you just need to install the proper JAX version.


## Skeleton
```
├── MCsampler/
│     ├── __init__.py
├── demo.ipynb
├── README.md
```

## Usage

You can find the main class wrapper inside the MCsampler main file. So far, only the Metropolis-Hastings algorithm is implemented. It can be called with `sampler = MCsampler.Metro_Has(P,D)`, where $P(x)$ is the target distribution and $D$ the number of dimensions. The sampler can be later called with `sampler.sample(x0, N)` to draw $~N$ samples from the initial state $x_0$. A `demo.ipynb` notebook with examples is provided.

For now it only works with $D=1$. Support for $D>1$ will be implemented soon.
