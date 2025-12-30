# lecture 1 notes

## micrograd overview

* micrograd is a tiny autograd (automatic differentiation) engine
* it implements backpropagation, which lets us iteratively adjust the weights in a neural network
* by tuning these weights, we can minimize a loss function

---

## derivative of a single function with one input

* consider the function:
  **f(x) = 3x² − 4x + 5**

* let **x = 3.0**

* then:
  **f(x) = f(3.0) = 20.0**

* choose a small value:
  **h = 0.001**

* evaluate the function slightly to the right:
  **f(x + h) = f(3.001) ≈ 20.01**

* compute the slope using the finite difference formula:

$$
\frac{f(x + h) - f(x)}{h}
$$

* this gives approximately **14.003**

* this means:

  * the slope at **x = 3.0** is positive
  * the derivative at that point is approximately **14.0**

---

## derivative of a function with multiple inputs

* consider the following values:

  * **a = 2.0**
  * **b = −3.0**
  * **c = 10.0**

* define:

  * **d = a · b + c**

* we want to know what happens when we take the derivative of **d** with respect to each input

* define:

  * **d₁ = a · b + c** (original value)
  * **d₂ = (a + h) · b + c**, or similarly for **b** or **c**

* the slope is computed as:

$$
\text{slope} = \frac{d_2 - d_1}{h}
$$

### intuition for each variable

* adjusting **a**:

  * since **b is negative**, increasing **a** makes **a · b** more negative
  * therefore, the slope with respect to **a** is negative

* adjusting **b**:

  * since **a is positive**, increasing **b** makes **a · b** less negative
  * therefore, the slope with respect to **b** is positive

* adjusting **c**:

  * **c** is added directly to **d**
  * increasing **c** increases **d** linearly
  * therefore, the slope with respect to **c** is positive

---

## the "value" object

the `value` class is the core building block of micrograd

each `value` object represents one number, but with extra information attached so we can later do backpropagation

you can think of a `value` as:

> “a number that remembers how it was made”

---

### `__init__`

```python
def __init__(self, data, _children=(), _op='', label=''):
````

* `data`
  the actual number (for example: 2.0, -3.5, etc)

* `grad`
  starts at `0.0`
  this will eventually store the gradient (how much this value affects the final output)

* `_prev`
  the values that were used to create this value
  this lets us build a computational graph

* `_op`
  the operation that created this value (`+`, `*`, etc)
  useful for debugging and graph visualization

* `label`
  an optional name for the value (purely for readability)

---

### `__repr__`

```python
def __repr__(self):
    return f"value(data={self.data})"
```

this controls how the object prints

instead of seeing something messy like `<object at 0x…>`,
you just see the number inside the value

---

### `__add__`

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    return out
```

this runs when you do:

```python
a + b
```

what happens:

* it adds the raw numbers: `self.data + other.data`
* it creates a new value object called `out`
* `out` remembers:

  * its parents (`self` and `other`)
  * the operation that created it (`'+'`)

so now the result knows where it came from

---

### `__mul__`

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    return out
```

this runs when you do:

```python
a * b
```

same idea as addition:

* multiply the numbers
* create a new value
* store:

  * the inputs
  * the operation (`'*'`)

again, the output remembers how it was built

---

## manual backprop

backpropagation tells us how each value contributes to the final output

each `value.grad` stores:

```
how much changing this value changes the final output L
```

in other words:

```
value.grad = dL / d(value)
```

---

### forward pass example

we define a small computation graph:

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b; e.label = 'e'
d = e + c; d.label = 'd'

f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
```

this corresponds to:

```
e = a * b
d = e + c
L = d * f
```

---

### base case for backprop

since the output is `L` itself:

```
dL / dL = 1
```

this is always true

so we start backprop by setting:

```python
L.grad = 1.0
```

this is the seed of the backward pass

---

### backprop through `L = d * f`

local derivatives:

```
dL / dd = f
dL / df = d
```

so we assign:

```python
d.grad = f
f.grad = d
```

this means:

* changing `d` affects `L` by a factor of `f`
* changing `f` affects `L` by a factor of `d`

---

### backprop through `d = e + c`

we apply the chain rule:

```
dL / de = (dL / dd) * (dd / de)
```

since:

```
dd / de = 1
dL / dd = f
```

we get:

```
dL / de = f
dL / dc = f
```

so:

```python
e.grad = f
c.grad = f
```

intuition:

* `e` and `c` both contribute equally to `d`
* whatever gradient flows into `d` flows unchanged into both

---

### backprop through `e = a * b`

again, use the chain rule

### gradient with respect to `a`

```
dL / da = (dL / de) * (de / da)
```

since:

```
de / da = b
dL / de = f
```

we get:

```
dL / da = b * f
```

### gradient with respect to `b`

by symmetry:

```
dL / db = a * f
```

so:

```python
a.grad = b * f
b.grad = a * f
```

---

### final gradients

after the full backward pass:

```
L.grad = 1
d.grad = f
f.grad = d
e.grad = f
c.grad = f
a.grad = b * f
b.grad = a * f
```

each value now knows exactly how it influenced the final output

---

## generalized backprop (formulas)

### multiplication

```
c = a * b
a.grad += b.data * c.grad
b.grad += a.data * c.grad
```

### addition

```
f = d + e
d.grad += f.grad
e.grad += f.grad
```

### mental rule

```
input.grad += (local derivative) * (output.grad)
```

---

## single neuron diagram

![single neuron diagram](https://www.cs.toronto.edu/~lczhang/360/lec/w02/imgs/neuron_model.jpeg)

* the diagram is showing **one neuron** (one “unit”) in a neural network
* it takes several inputs, combines them, then pushes the result through an activation function

---

### what each part means

* inputs (`x0`, `x1`, `x2`, ...)

  * each `x_i` is an input number coming into the neuron
  * these can be:

    * raw features from data
    * outputs from neurons in the previous layer

* weights (`w0`, `w1`, `w2`, ...)

  * each input has a weight `w_i`
  * the weight controls how strongly that input matters
  * each input contributes:

    * `w_i * x_i`

* weighted sum + bias

  * the neuron adds up all the weighted inputs:

    * `Σ (w_i * x_i)`

  * the neuron also adds a bias term:

    * `+ b`

  * bias lets the neuron shift the whole sum up or down
  * it’s like a “default offset” even if inputs are zero

* pre-activation value (`z`)

  * the thing before the activation function is usually called `z`

    * `z = Σ (w_i * x_i) + b`

  * this is the **linear** part of the neuron

* activation function (`f`)

  * the neuron then applies a function `f` to `z`

    * `output = f(z)`

  * this is what makes the neuron **nonlinear**
  * without this, stacking layers wouldn’t add any real power

* output axon

  * the output `f(z)` is sent forward as the neuron’s final output
  * this becomes an input to later neurons

---

## why `tanh` is a good activation function

* a common choice is:

  * `f(z) = tanh(z)` because it always stays between `-1` and `1`

* outputs are **bounded**
* values don’t blow up as they pass through layers
* the neuron can output:

  * negative values
  * zero
  * positive values
