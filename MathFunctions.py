import math

def getRotationAngle(p1, p2, ref_vec, x_ind, y_ind):

	"""Function to compute rotation angle of a vector relative to a reference vector
	
	Arguments:
	----------
	
	p1			a 2D point set
	p2			a 2D point set
	ref_vec		a reference vector
	x_ind		an index into a point set for x
	y_ind		an index into a point set for y
	
	Returns:
	--------
	
	An angle
	"""
	
	# obtain the vector from two points
	vec = computeVectorFromPoints([p1[x_ind], p1[y_ind]], [p2[x_ind], p2[y_ind]])
	
	# obtain the vector magnitude
	mag = computeVectorMagnitude(vec)
	
	# normalise the vector
	norm = normalizeVector(vec, mag)
	
	# obtain the nagle between the normalised and reference vectors
	angle = computeVectorAngle(norm, ref_vec)
	
	# correct the sign of the angle
	if norm[1] > 0: angle = -angle
	
	# return the result
	return angle
def computeVectorMagnitude(vec):

	return math.sqrt((vec[0]*vec[0])+(vec[1]*vec[1]))


def computeVectorFromPoints(p2, p1):

	"""Function to construct a vector from two points
	
	Arguments:
	----------
	
	p2	a 2D point
	p1	a 2D point
	
	Returns:
	--------
	
	A 2D vector
	"""
	
	# return the result
	return [p2[0] - p1[0], p2[1] - p1[1]]
		
def getRotationAngleFromProfile(profile, ref_vec, x_ind, y_ind):

	"""Method to compute corrective rotation angle for aligning a (nod)

	Arguments:
	----------

	profile		a set of 2D points that describe 
	ref_vec		a reference vector
	x_ind		an index into a point set for x
	y_ind		an index into a point set for y

	Returns:
	--------

	An angle
	"""	

	# extract the x values from the profile
	x_vals = [i[x_ind] for i in profile]

	# extract the ya values from the profile
	y_vals = [i[y_ind] for i in profile]

	# obtain a covariance matrix for the x and y values
	cov_mat = computeCovarianceMatrix(x_vals, y_vals)

	# obtain the eigenvalues of the covariance matrix
	eig_vals = computeEigenValues(cov_mat)

	# obtain the eigenvector of the covariance matrix
	eig_vec = computeEigenVector(cov_mat, eig_vals[0], eig_vals[1])

	# compute the angle between the eigenvector and the reference vector
	angle = computeVectorAngle(eig_vec, ref_vec)

	# correct the sign of the angle
	if eig_vec[1] > 0: angle = -angle

	# return the result
	return angle





def normalizeVector(vec, mag):

	"""Function to normalize a vector
	
	Arguments:
	----------
	
	vec		a 2D vector
	mag		the vector's magnitude
	
	Returns:
	--------
	
	A 2D vector
	"""

	# compute normalised x component
	x = vec[0] / mag
	
	# compute normalised y component
	y = vec[1] / mag
	
	# return the result
	return [x, y]

def computeVectorAngle(v1, v2):

	"""Function to compute the angle between two vectors in radians
	
	Arguments:
	----------
	
	v1	a 2D vector
	v2	a 2D vector
	
	Returns:
	--------
	
	An angle in radians
	"""
	
	# compute the dot product of the vectors
	dot_prod = v1[0] * v2[0] + v1[1] * v2[1]
	
	# compute the angle
	angle = math.acos(dot_prod)
	
	# return the result
	return angle

def computeCovarianceMatrix(x_vals, y_vals):

	"""Function to compute a covariance matrix from a list of x and y values
	
	Arguments:
	----------
	
	x_vals	a list of x values
	y_vals	a list of y values
	
	Returns:
	--------
	
	A covariance matrix
	"""

	# convert x values to Python list
	x = list(x_vals)
	
	# convert y values to Python list
	y = list(y_vals)

	# compute mean of x values
	mean_x = float(sum(x)/len(x))
	
	# compute mean of y values
	mean_y = float(sum(y)/len(y))

	# compute variance across x
	cov_xx = sum([((x[i]-mean_x) * (x[i]-mean_x)) for i in range(len(x))]) / len(x)
	
	# compute variance across y
	cov_yy = sum([((y[i]-mean_y) * (y[i]-mean_y)) for i in range(len(x))]) / len(x)
	
	# compute variance across x and y
	cov_xy = sum([((x[i]-mean_x) * (y[i]-mean_y)) for i in range(len(x))]) / len(x)

	# format the covariance matrix
	cov_mat = [[cov_xx, cov_xy], [cov_xy, cov_yy]]

	# return the result
	return cov_mat

def computeEigenValues(mat):

	"""Function to compute the eigenvalues of a 2x2 matrix
	
	Arguments:
	----------
	
	mat	a 2 x 2 matrix
	
	Returns:
	--------
	
	A list of eigenvalues
	"""

	# extract the values in the matrix
	M11 = mat[0][0]
	M12 = mat[0][1]
	M21 = mat[1][0]
	M22 = mat[1][1]

	# Compute the trace of the matrix
	tr_M = M11 + M22
	
	# compute the determinant of the matrix
	det_M = (M11 * M22) - (M12 * M21)

	# Declare coefficients of quadratic
	a = 1
	b = -tr_M
	c = det_M

	# Solve the pair of quadratics to compute the eigenvalues of the matrix
	x1 = (-b + (math.sqrt((b * b) - (4 * a * c)))) / (2 * a)
	x2 = (-b - (math.sqrt((b * b) - (4 * a * c)))) / (2 * a)

	# return the result
	return [x1, x2]

def computeEigenVector(mat, L1, L2):

	"""Function to compute the eigenvector associated with the largest eigenvalue of a matrix
	
	Arguments:
	----------
	
	mat	a 2 x 2 matrix
	L1	possible value of Lambda
	L2	possible value of Lambda
	
	Returns:
	--------
	
	A normalised eigenvector
	"""

	# Set Lambda initially to L1
	L = L1
	
	# if L2 is the larger, set Lambda to L2
	if L2 > L1: L = L2

	# Set x component of vector equal to 1
	x = 1

	# extract the values in the matrix
	M11 = mat[0][0]
	M12 = mat[0][1]
	M21 = mat[1][0]
	M22 = mat[1][1]

	# y component of vector is computed using (A-(Lambda * I)v = 0, where I is the identity matrix
	y = -(M11 - L) / M12
	
	# format the vector as a Python list
	vec = [x, y]

	# obtain the vector magitude
	mag = computeVectorMagnitude(vec)
	
	# normalise the vector
	norm = normalizeVector(vec, mag)
	
	# return the normalised vector
	return norm















