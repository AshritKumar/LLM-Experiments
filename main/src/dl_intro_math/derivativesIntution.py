# at its core derivative of a function y = f(x) at some point is if we slightly bump up x by `h`, how will the function respond - this is the essence of derivative
# so its just (y2-y1)/(x2 - x1), which is the slope of the secant line between two points
# now, if we let h approach 0, we get the slope of the tangent line at that point, which is the derivative

# so, say y1 = f(x), y2 = f(x+h), then the difference quotient is (f(x+h) - f(x))/(x+h - x) = (f(x+h) - f(x))/h
# this way we can compute the derivative of a function at any point


def fx(x):
    return 3*x**2 - 4*x + 5 # 3x^2 - 4x + 5, parabolic eq


# manual derivative when we have one input 'x'
x = 3
h = 0.0000001 
dx = (fx(x+h) - fx(x)) / h

print(f"At x={x}, h={h}, derivative approximation = {dx:.6f}")


# This is a function with 3 inputs, we need partial derivatives here
def fxyz(x,y,z):
    u = x*y + z
    return u

# Manual partial derivatives calc
x = 2
y = -3
z = 10

u = fxyz(x, y, z)
print(f"fxyz({x}, {y}, {z}) = {u}")

# derivative of the function u w.r.t 'x' increase x by some small amount h and keep y,z constant
x1 = x + h
u1x = fxyz(x1, y, z)
print(f"u1x = fxyz({x1}, {y}, {z}) = {u1x}")
# partial derivative w.r.t x: how u changes when we change x by h, keeping y,z constant
dux = (u1x - u) / h
print(f"Partial derivative of u w.r.t x: {dux:.6f}")


