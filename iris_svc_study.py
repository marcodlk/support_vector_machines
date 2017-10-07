from sklearn.datasets import load_iris
import numpy as np
import svmlab

# load iris dataset
iris = load_iris()
x_data = iris.data[:, :2] # only first two features
y_data = iris.target
classes = iris.target_names

# only 2 classes
#x_data = x_data[y_data != 2]
#y_data = y_data[y_data != 2]
#classes = classes[:2]

# interval in seconds, for animations
INTERVAL = 0.5

# output directory name
ANIM_DIR = 'animations/'

# SVC filenames
svc_linear_C_fname = ANIM_DIR + 'iris_svc_linear_C.mp4'
svc_poly_C_fname = ANIM_DIR + 'iris_svc_poly_C.mp4'
svc_poly_gamma_fname = ANIM_DIR + 'iris_svc_poly_gamma.mp4'
svc_rbf_C_fname = ANIM_DIR + 'iris_svc_rbf_C.mp4'
svc_rbf_gamma_fname = ANIM_DIR + 'iris_svc_rbf_gamma.mp4'
svc_rbf_dual_fname = ANIM_DIR + 'iris_svc_rbf_dual.mp4'

# NuSVC filenames
nusvc_linear_nu_fname = ANIM_DIR + 'iris_nusvc_linear_nu.mp4'

# generate C, nu, and gamma static lists
static_C_list = [1.0 for _ in range(30)]
static_nu_list = [0.5 for _ in range(30)]
static_gamma_list = [1.0 for _ in range(30)]

# generate C, nu, and gamma dynamic lists
dynamic_C_list = np.logspace(-2, 3, 30)
dynamic_nu_list = np.linspace(0., 0.9, 50)[1:-1] # 0 and 1 are infeasible nu values, strip **
dynamic_gamma_list = np.logspace(-2, 3, 30)

# ** note it seems that actually nu values greater than 0.9 are infeasible

# initialize svm student with iris dataset
print('-- initializing support vector machines student ')
student = svmlab.Student(x_data, y_data)

# svc, linear kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on SVC, linear kernel ')
student.study_svc(kernel='linear', C_list=dynamic_C_list)
student.publish(svc_linear_C_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_linear_C_fname)

# svc, poly kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on SVC, poly kernel ')
student.study_svc(kernel='poly', 
                  C_list=dynamic_C_list,
                  gamma_list=static_gamma_list)
student.publish(svc_poly_C_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_poly_C_fname)

# svc, rbf kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on SVC, rbf kernel ')
student.study_svc(kernel='rbf', 
                  C_list=dynamic_C_list,
                  gamma_list=static_gamma_list)
student.publish(svc_rbf_C_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_rbf_C_fname)

# svc, rbf kernel, study effect of gamma parameter (kernel coefficient)
print('-- studying effect of gamma parameter on SVC, rbf kernel ')
student.study_svc(kernel='rbf', 
                  C_list=static_C_list,
                  gamma_list=dynamic_gamma_list)
student.publish(svc_rbf_gamma_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_rbf_gamma_fname)

# svc, rbf kernel, study effect of both C and gamma parameters (jointly incrementing)
print('-- studying effect of both C and gamma parameters on SVC, rbf kernel ')
student.study_svc(kernel='rbf', 
                  C_list=dynamic_C_list,
                  gamma_list=dynamic_gamma_list)
student.publish(svc_rbf_dual_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_rbf_dual_fname)

# nusvc, linear kernel, study effect of nu parameter
print('-- studying effect of nu parameter on NuSVC, linear kernel ')
student.study_nusvc(kernel='linear', nu_list=dynamic_nu_list)
student.publish(nusvc_linear_nu_fname, interval=INTERVAL)
print('   ... saved to file %s ' % nusvc_linear_nu_fname)

print('-- done ')
