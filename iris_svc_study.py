from sklearn.datasets import load_iris
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
ANIM_DIR = 'animations/'
svc_linear_C_fname = ANIM_DIR + 'iris_svc_linear_C.mp4'
svc_poly_C_fname = ANIM_DIR + 'iris_svc_poly_C.mp4'
svc_poly_gamma_fname = ANIM_DIR + 'iris_svc_poly_gamma.mp4'
svc_rbf_C_fname = ANIM_DIR + 'iris_svc_rbf_C.mp4'
svc_rbf_gamma_fname = ANIM_DIR + 'iris_svc_rbf_gamma.mp4'
svc_rbf_dual_fname = ANIM_DIR + 'iris_svc_rbf_dual.mp4'

# initialize svc with iris dataset
print('-- initializing support vector classifier (SVC) student ')
svc_student = svmlab.SVCStudent(x_data, y_data)

# svc, linear kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on linear kernel ')
svc_student.study_kernel('linear', dynamic_C=True)
svc_student.visualize_study(svc_linear_C_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_linear_C_fname)

# svc, poly kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on poly kernel ')
svc_student.study_kernel('poly', dynamic_C=True)
svc_student.visualize_study(svc_poly_C_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_poly_C_fname)

# svc, rbf kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on rbf kernel ')
svc_student.study_kernel('rbf', dynamic_C=True)
svc_student.visualize_study(svc_rbf_C_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_rbf_C_fname)

# svc, rbf kernel, study effect of gamma parameter (kernel coefficient)
print('-- studying effect of gamma parameter on rbf kernel ')
svc_student.study_kernel('rbf', dynamic_gamma=True)
svc_student.visualize_study(svc_rbf_gamma_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_rbf_gamma_fname)

# svc, rbf kernel, study effect of both C and gamma parameters (jointly incrementing)
print('-- studying effect of both C and gamma parameters on rbf kernel ')
svc_student.study_kernel('rbf', dynamic_C=True, dynamic_gamma=True)
svc_student.visualize_study(svc_rbf_dual_fname, interval=INTERVAL)
print('   ... saved to file %s ' % svc_rbf_dual_fname)

print('-- done ')
