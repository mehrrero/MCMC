import jax
import jax.numpy as jnp
from jax import jit
from jax.lib import xla_bridge
from jax.lax import scan
from jax import random
import timeit


@jit
def normal(xp,x):
    '''
        Evaluates a normal distribution with sigma =1 and centered at xp.

        Parameters:
            xp (float): Center of the normal distribution.
            x (float): Point where to evaluate the distribution.

        Returns:
            n (float): Probability of point x

    '''
    
    return  1/jnp.sqrt(2*jnp.pi)*jnp.exp(-1/2*(x-xp)**2)


def multi_normal(xp, x, D):
    '''
        Evaluates a multivariate normal distribution with sigma = 1/2 * Identity and centered at xp.

        Parameters:
            xp (array): Center of the distribution.
            x (array): Point where to evaluate the distribution.

        Returns:
            n (float): Probability of point x

    '''

    cov = 1/2*jax.numpy.identity(D)
    exponent = -1/2 * jnp.matmul(jnp.matmul(jnp.transpose((xp - x)), cov), (xp - x))
    return 1/(2 * jnp.pi) ** (D / 2) * jnp.exp(exponent)



class Metro_Has():
    '''
        Implementation of the Metropolis-Hastings algorithm to sample from a D-dimensional distribution.

        Parameters:
            P (function): Distribution to be sampled.
            D (int): Dimensionality of base space.
        
    '''

    def __init__(self, P, D):
        super(Metro_Has, self).__init__()
        self.P = P
        self.key = random.PRNGKey(0)
        self.D = D

    def draw_sample(self,x, key):
        '''
            Draws a sample from the proposal distribution, chosen to be a (multivariate) normal distribution with unit covariance.

            Parameters:
                x (array): Current state of the Markov chain.
                key (array): PRNG key for the random generators.

            Returns:
                xp (array): Sample from the proposal distribution.

        '''
        
        if self.D == 1:
            xp = jax.random.normal(key)

        if self.D >1:
            xp = jax.random.multivariate_normal(key, x, jax.numpy.identity(self.D))
        return xp

    
    def Q(self, xp, x):
        '''
            Probability function of the proposal distribution, chosen to be a (multivariate) normal distribution with unit covariance.
        
            Parameters:
                xp (array): New proposed state of the Markov chain.
                x (array): Current state of the Markov chain.

            Returns:
                q (float): Probability of Q(xp|x).
        '''
        
        if self.D == 1:
            q = normal(xp,x)
            
        elif self.D >1:
            q = multi_normal(xp,x, self.D)

        return q
        
    
    def acceptance(self, x, xp, key):
        '''
            Checks whether we accept or not the new proposed state in the Markov chain.

            Parameters:
                x (array): Current state of the Markov chain.
                xp (array): New proposed state of the Markov chain.
                key (array): PRNG key for the random generators.
                
            Returns:
                xp (array): Returns the value of xp when it is accepted. Otherwise it returns nan.
            
        '''
        alpha = [1]
        alpha.append(self.P(xp)/self.P(x) * self.Q(x,xp)/self.Q(xp,x) )
        a = jnp.amin(jnp.array(alpha))
        u = random.uniform(key)
        xp = jnp.where(u > a, jnp.nan, xp)
        
        return xp

    
    
    def step(self, x0, key1, key2):
        '''
            Performs an attemp to a step in the Markov chain.
            
            Parameters:
                x0 (array): Current state of the Markov chain.
                key1 (array): PRNG key for the random generators.
                key2 (array): PRNG key for the random generators.

            Returns:
                xp (array): Returns the value of xp when it is accepted. Otherwise it returns nan.
        '''
        

        xp = self.draw_sample(x0, key1)
        
        return self.acceptance(x0, xp, key2)


    def markov_chain(self, x0, key, N, burning = 50):
        '''
            Performs N attempts of moving along the Markov chain, with N the lenght of keys.

            Parameters:
                x0 (array): Current state of the Markov chain.
                keys (array): PRNG keys for the random generators.
                burning_keys (array): PRNG keys for the random generators used during the burning period.
                
            Returns:
                x_ac (array): Accepted values.
                
        '''
        
        
        x_ac = []

        for i in range(burning):
            key, subkey1, subkey2 = random.split(key, 3)
            x_n = self.step(x0, subkey1, subkey2)
            x0 = jnp.where(~jnp.any(jnp.isnan(x_n)), x_n, x0)
        
        for i in range(N):
            key, subkey1, subkey2 = random.split(key, 3)
            x_n = self.step(x0,subkey1, subkey2) 
            x0 = jnp.where(~jnp.any(jnp.isnan(x_n)), x_n, x0)
            x_ac.append(x_n)

        x_ac = jnp.stack(x_ac)

        return x_ac


    
    def sample(self, x0, Nt, burning = 50):
        '''
            Returns ~Nt samples from the distribution.

            Parameters:
                x0 (array): Initial state.
                Nt (int): Approximate number of samples to be retrieved from the distribution. The sampling will always return a number close to but larger than Nt.
                burn (int, optional): Number of steps in the burning stage. The default value is 50.

            Returns:
                u (array): ~Nt samples of the target distribution.
        '''
        
        Nt = int(Nt)
        N = 20

        Nc = Nt//N


        start = timeit.default_timer()
        u = jnp.array([])
        
        while len(u) < self.D*Nt:
            keys = random.split(self.key, Nc + 1)
            self.key, *keys = keys
            keys = jnp.array(keys)
            
            _x0 = jnp.expand_dims(x0,0).repeat(Nc, axis=0)
            u_p = jax.vmap(lambda x, y: self.markov_chain(x, y, N, burning = burning))(_x0,keys)
            u_p = u_p[~jnp.isnan(u_p)]
            u = jnp.concatenate((u,u_p))
            x0 = u[-self.D:]
            if self.D == 1:
                x0 = x0.item()

        stop = timeit.default_timer()
        dt = (stop - start) 
        
        if self.D>1:
            u = u.reshape(u.shape[0]//self.D,self.D)
        

        print(f'{len(u)} points accepted in {dt} s.')
        return u





