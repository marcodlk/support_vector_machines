from sklearn.datasets import load_iris
from matplotlib import colors
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
HEAT_DIR = 'heatmaps/'

# SVC filenames
svc_linear_C_fname = ANIM_DIR + 'iris_svc_linear_C.mp4'
svc_poly_C_fname = ANIM_DIR + 'iris_svc_poly_C.mp4'
svc_poly_gamma_fname = ANIM_DIR + 'iris_svc_poly_gamma.mp4'
svc_rbf_C_fname = ANIM_DIR + 'iris_svc_rbf_C.mp4'
svc_rbf_gamma_fname = ANIM_DIR + 'iris_svc_rbf_gamma.mp4'
svc_rbf_both_fname = ANIM_DIR + 'iris_svc_rbf_both.mp4'
# heatmaps
svc_linear_grid_search = HEAT_DIR + 'iris_svc_linear_grid_search.png'
svc_poly_grid_search = HEAT_DIR + 'iris_svc_poly_grid_search.png'
svc_rbf_grid_search = HEAT_DIR + 'iris_svc_rbf_grid_search.png'

# NuSVC filenames
nusvc_linear_nu_fname = ANIM_DIR + 'iris_nusvc_linear_nu.mp4'
nusvc_poly_nu_fname = ANIM_DIR + 'iris_nusvc_poly_nu.mp4'
nusvc_poly_gamma_fname = ANIM_DIR + 'iris_nusvc_poly_gamma.mp4'
nusvc_rbf_nu_fname = ANIM_DIR + 'iris_nusvc_rbf_nu.mp4'
nusvc_rbf_gamma_fname = ANIM_DIR + 'iris_svc_rbf_gamma.mp4'
# heatmaps
nusvc_linear_grid_search = HEAT_DIR + 'iris_nusvc_linear_grid_search.png'
nusvc_poly_grid_search = HEAT_DIR + 'iris_nusvc_poly_grid_search.png'
nusvc_rbf_grid_search = HEAT_DIR + 'iris_nusvc_rbf_grid_search.png'

# generate C, nu, and gamma static lists
static_C_range = [1.0 for _ in range(30)]
static_nu_range = [0.5 for _ in range(30)]
static_gamma_range = [1.0 for _ in range(30)]

# generate C, nu, and gamma dynamic lists
dynamic_C_range = np.logspace(-2, 3, 30)
dynamic_nu_range = np.linspace(0., 0.9, 50)[1:-1] # 0 and 1 are infeasible nu values, strip **
dynamic_gamma_range = np.logspace(-2, 3, 30)

# ** note it seems that actually nu values greater than 0.9 are infeasible

# initialize svm lab with iris dataset
print('-- initializing support vector machines lab')
lab = svmlab.SVMLab(x_data, y_data)

# svc, linear kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on SVC, linear kernel ')
lab.svm_animation(
    svm_=dict(impl='svc', kernel='linear', dfs='ovr'),
    range_=dict(C=dynamic_C_range, gamma=None),
    animation_=dict(filename=svc_linear_C_fname)
)
print('   ... saved to file %s ' % svc_linear_C_fname)

# svc, poly kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on SVC, poly kernel ')
lab.svm_animation(
    svm_=dict(impl='svc', kernel='poly', dfs='ovr'),
    range_=dict(C=dynamic_C_range, gamma=static_gamma_range),
    animation_=dict(filename=svc_poly_C_fname)
)
print('   ... saved to file %s ' % svc_poly_C_fname)

# svc, rbf kernel, study effect of C parameter (softmargin)
print('-- studying effect of C parameter on SVC, rbf kernel ')
lab.svm_animation(
    svm_=dict(impl='svc', kernel='rbf', dfs='ovr'),
    range_=dict(C=dynamic_C_range, gamma=static_gamma_range),
    animation_=dict(filename=svc_rbf_C_fname)
)
print('   ... saved to file %s ' % svc_rbf_C_fname)

# svc, rbf kernel, study effect of gamma parameter (kernel coefficient)
print('-- studying effect of gamma parameter on SVC, rbf kernel ')
lab.svm_animation(
    svm_=dict(impl='svc', kernel='rbf', dfs='ovr'),
    range_=dict(C=static_C_range, gamma=dynamic_gamma_range),
    animation_=dict(filename=svc_rbf_gamma_fname)
)
print('   ... saved to file %s ' % svc_rbf_gamma_fname)

# svc, rbf kernel, study effect of both C and gamma parameters (jointly incrementing)
print('-- studying effect of both C and gamma parameters on SVC, rbf kernel ')
lab.svm_animation(
    svm_=dict(impl='svc', kernel='rbf', dfs='ovr'),
    range_=dict(C=dynamic_C_range, gamma=dynamic_gamma_range),
    animation_=dict(filename=svc_rbf_both_fname)
)
print('   ... saved to file %s ' % svc_rbf_both_fname)

# nusvc, linear kernel, study effect of nu parameter
print('-- studying effect of nu parameter on NuSVC, linear kernel ')
lab.svm_animation(
    svm_=dict(impl='nusvc', kernel='linear', dfs='ovr'),
    range_=dict(nu=dynamic_nu_range, gamma=None),
    animation_=dict(filename=nusvc_linear_nu_fname)
)
print('   ... saved to file %s ' % nusvc_linear_nu_fname)
   
#-------------------------------------------------------------------------------
# svc, grid search with cross validation to find optimal parameters
# linear
C_range = np.logspace(-3, 9, 13)
gamma_range = [0.0] # no gamma in linear
print('-- svc, linear kernel, grid search for optimal parameters ')
best_params, best_score = lab.optimal_param_grid_search(
    svm_=dict(impl='svc', kernel='linear', dfs='ovr'),
    range_=dict(C=C_range, gamma=gamma_range), 
    heatmap_=dict(
        filename=svc_linear_grid_search,
        norm=colors.PowerNorm(gamma=5.)
    )
)
print('   * the best parameters are %s with a score of %0.2f'
          % (best_params, best_score))
print('   ... saved to file %s ' % svc_linear_grid_search)

''' **disabled because poly takes way too long **
# poly 
C_range = np.logspace(-2, 2, 5)
gamma_range = np.logspace(-2, 2, 5)
print('-- svc, poly kernel, grid search for optimal parameters ')
best_params, best_score = lab.optimal_param_grid_search(
    svm_=dict(impl='svc', kernel='poly', dfs='ovr'),
    range_=dict(C=C_range, gamma=gamma_range), 
    heatmap_=dict(filename=svc_poly_grid_search)
)
print('   * the best parameters are %s with a score of %0.2f'
          % (best_params, best_score))
print('   ... saved to file %s ' % svc_poly_grid_search)
'''

# rbf
#C_range = np.logspace(-3, 3, 7)
#gamma_range = np.logspace(-3, 3, 7)
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
print('-- svc, rbf kernel, grid search for optimal parameters ')
best_params, best_score = lab.optimal_param_grid_search(
    svm_=dict(impl='svc', kernel='rbf', dfs='ovr'),
    range_=dict(C=C_range, gamma=gamma_range), 
    heatmap_=dict(filename=svc_rbf_grid_search)
)
print('   * the best parameters are %s with a score of %0.2f'
          % (best_params, best_score))
print('   ... saved to file %s ' % svc_rbf_grid_search)

# nusvc, grid search with cross validation to find optimal parameters
#lab.optimal_param_grid_search('nusvc', nu_range, gamma_range,
#                              heatmap='nusvc_optimal_param_search.png')

print('-- done ')
