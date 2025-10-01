# rl-mujoco

do not mess with gradients directly - use the standard optimisers otherwise not even SDG might work...actually, maybe? the same code, same parameters, sometimes just converge, some times just dont...
this is probably a mixture of things, and neural network initializations could be part of that. full zero init seems bad, but you need to set something for the last layers to start ruoghly right. so perhaps start with a high bias at the end
overall you would like to have roughly random init across a range of reasonable starting states, to ensure exploraiton, but yeah thats brittle if its your only mechanism
add epsilons wherever an operation has specific bounds, divisions etc

note that the action space is a vector, and i model the 3 components separately. however the probability is to the action vector, so i have to multiply them together.
also the modifier works this way as you would have the jacobian, but you apply the tanh separately meaning that the cross partial derivatives (dx/dy) are null, hence the matrix is diagonal and the determinant is the product of the derivatives which in log is their sum.