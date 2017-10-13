import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm

KERNELS_WITH_GAMMA = ['rbf', 'poly', 'sigmoid']

''' Fit classifier and collect useful data and metadata
'''
def run_classifier(clf, x_train, y_train, x_test=None, y_test=None):
    # fit classifier according to training data
    clf.fit(x_train, y_train)
    
    # make predictions over a mesh grid to visualize classification regions
    x_min, x_max = x_train[:,0].min() - 0.5, x_train[:,0].max() + 0.5
    y_min, y_max = x_train[:,1].min() - 0.5, x_train[:,1].max() + 0.5
    step = .02
    XX, YY = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    # if classification is binary, use decision function
#    n_classes = y_train.max() - y_train.min() + 1
#    if n_classes == 2:
#        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
#    else:
#        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    
    if x_test and y_test:
        # score the fit
        score = clf.score(x_test, y_test)

        # confusion matrix
        y_pred = clf.predict(x_test)
        cnf_mat = confusion_matrix(y_test, y_pred)
        # normalize
        cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

    else:
        score = 'none'
        cnf_mat = 'none'

    # extract support vectors
    x_support = clf.support_vectors_
    
    # compile results and necessary info
    results = {
        'metadata': clf.__dict__,
        'training set': (x_train, y_train),
        'testing set': (x_test, y_test),
        'support': (x_support),
        'confusion': cnf_mat, 
        'contour': (XX, YY, Z),
        'score': score,
        'bounds': (x_min, x_max, y_min, y_max)
    }
    
    return results, clf

def test_svm(x_train, y_train, x_test, y_test, 
             svm_=dict(impl='svc', kernel='rbf', C=1.0, nu=0.5, gamma=1.0, dfs='ovr')):
    kernel, dfs = svm_['kernel'], svm_['dfs']
    if svm_['impl'] in ['svc', 'SVC', 'c_svc']:
        C, gamma = svm_['C'], svm_['gamma']
        if kernel not in KERNELS_WITH_GAMMA:
            clf = svm.SVC(kernel=kernel, C=C, decision_function_shape=dfs)
        else:
            clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, decision_function_shape=dfs)
    elif svm_['impl'] in ['nusvc', 'NuSVC', 'nu_svc']:
        nu, gamma = svm_['nu'], svm_['gamma']
        if kernel not in KERNELS_WITH_GAMMA:
            clf = svm.NuSVC(kernel=kernel, nu=nu, decision_function_shape=dfs)
        else:
            clf = svm.NuSVC(kernel=kernel, nu=nu, gamma=gamma, decision_function_shape=dfs)
    else:
        print('svmlab: Unrecognized svm type')
        assert False

    # fit classifier on training data
    clf.fit(x_train, y_train)
    
    # gather useful classifier info after fit
    n_supports = len(clf.support_vectors_)

    # score classifier on testing data
    score = clf.score(x_test, y_test)

    return score, n_supports
    

def develop_svc_results(x_data, y_data, kernel, C, gamma, dfs='ovr'):
    if kernel not in KERNELS_WITH_GAMMA:
        clf = svm.SVC(kernel=kernel, C=C, decision_function_shape=dfs)
    else:
        clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, decision_function_shape=dfs)

    results, _ = run_classifier(clf, x_data, y_data)

    return results

def develop_multiple_svc_results(x_data, y_data, kernel, C_range, gamma_range, dfs='ovr'):
    # initialize results list
    results_buffer = []

    # if kernel does not use gamma, create arbitrary gamma_range to zip with C_range
    if kernel not in KERNELS_WITH_GAMMA: 
        gamma_range = [0.0 for _ in range(len(C_range))]

    # C and gamma lists should have same length
    assert len(C_range) == len(gamma_range)
    
    # iterate through C, gamma tuples and develop results    
    for C, gamma in zip(C_range, gamma_range):
        results_buffer.append(develop_svc_results(x_data, y_data, kernel, C, gamma, dfs=dfs))

    return results_buffer

def develop_nusvc_results(x_data, y_data, kernel, nu, gamma, dfs='ovr'):
    if kernel not in KERNELS_WITH_GAMMA:
        clf = svm.NuSVC(kernel=kernel, nu=nu, decision_function_shape=dfs)
    else:
        clf = svm.NuSVC(kernel=kernel, nu=nu, gamma=gamma, decision_function_shape=dfs)

    results, _ = run_classifier(clf, x_data, y_data)

    return results

def develop_multiple_nusvc_results(x_data, y_data, kernel, nu_range, gamma_range, dfs='ovr'):
    # initialize resultslist
    results_buffer = []

    # if kernel does not use gamma, create arbitrary gamma_range to zip with nu_range
    if kernel not in KERNELS_WITH_GAMMA: 
        gamma_range = [0.0 for _ in range(len(nu_range))]

    # nu and gamma lists should have same length
    assert len(nu_range) == len(gamma_range)

    # iterate through nu, gamma tuples and develop results
    for nu, gamma in zip(nu_range, gamma_range):
        if nu > 0.9:
            print('svmlab: study cut short due to infeasible nu value (greater than 0.9)')
            break
        results_buffer.append(develop_nusvc_results(x_data, y_data, kernel, nu, gamma, dfs=dfs))

    return results_buffer
    
''' Support vector machines student class
'''
class SVMLab:
    def __init__(self, x_data, y_data):
        self.x_data, self.y_data = x_data, y_data
        self.results_buffer = []
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111)

    def assign_new_data(self, x_data, y_data):
        self.x_data, self.y_data = x_data, y_data

    def standard_scale_features(self):
        self.x_data = StandardScaler().fit_transform(self.x_data)

    def svm_test(self, x_test, y_test, 
                 svm_=dict(impl='svc', kernel='rbf', C=1.0, nu=0.5, gamma=1.0, dfs='ovr')):
        return test_svm(self.x_data, self.y_data, x_test, y_test, svm_)
        

    def svm_animation(self, 
                      svm_=dict(impl='svc', kernel='rbf', dfs='ovr'),
                      range_=dict(C=[1.], nu=[1.], gamma=[1.]),
                      animation_=dict(filename='svm_animation.mp4', interval=0.5)):

        if svm_['impl'] in ['svc', 'SVC', 'c_svc']:
            self.results_buffer = develop_multiple_svc_results(
                self.x_data, self.y_data, 
                svm_['kernel'], range_['C'], range_['gamma'], dfs=svm_['dfs']
            )
        elif svm_['impl'] in ['nusvc', 'NuSVC', 'nu_svc']:
            self.results_buffer = develop_multiple_nusvc_results(
                self.x_data, self.y_data, 
                svm_['kernel'], range_['nu'], range_['gamma'], dfs=svm_['dfs']
            )
        else:
            print('svmlab: Unrecognized svm type')
            assert False

        interval = 0.5
        if 'interval' in animation_:
            interval = animation_['interval'] #in seconds
        anim = FuncAnimation(self.fig, self.draw_plot, self.report,
                             interval=interval*1e+3, blit=False)
        anim.save(animation_['filename'])
        self.fig_reset()

        return

    def svm_plot(self, 
                 svm_=dict(impl='svc', kernel='rbf', C=1.0, gamma='auto', dfs='ovr'),
                 plot_=dict(filename='svm_plot.png')):
        
        if svm_['impl'] in ['svc', 'SVC', 'c_svc']:
            self.draw_plot(
                develop_svc_results(
                    self.x_data, self.y_data,
                    svm_['kernel'], svm_['C'], svm_['gamma'], dfs=svm_['dfs']
                )
            )
        elif svm_['impl'] in ['nusvc', 'NuSVC', 'nu_svc']:
            self.draw_plot(
                develop_nusvc_results(
                    self.x_data, self.y_data,
                    svm_['kernel'], svm_['nu'], svm_['gamma'], dfs=svm_['dfs']
                )
            )
        else:
            print('svmlab: Unrecognized svm type')
            assert False

        self.fig.savefig(plot_['filename'])
        self.fig_reset()

        return

    def optimal_param_grid_search(self, 
                                  svm_=dict(impl='svc', kernel='rbf', dfs='ovr'),
                                  range_=dict(C=[1.], nu=[1.], gamma=[1.]),
                                  heatmap_=dict(
                                    filename='svm_gridsearch.png',
                                    norm=None
                                  )):
        # build the defined svm
        svm_obj = None
        param_grid = dict(gamma=range_['gamma'])
        if svm_['impl'] in ['svc', 'SVC', 'c_svc']:
            svm_obj = svm.SVC(kernel=svm_['kernel'], decision_function_shape=svm_['dfs'])
            param_grid['C'] = range_['C']
        elif svm_['impl'] in ['nusvc', 'NuSVC', 'nu_svc']:
            svm_obj = svm.NuSVC(kernel=svm_['kernel'], decision_function_shape=svm_['dfs'])
            param_grid['nu'] = range_['nu']
        else:
            print('svmlab: Unrecognized svm type')
            assert False

        # cross validation strategy
        cross_validation = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        
        # perform grid search
        grid = GridSearchCV(svm_obj, param_grid=param_grid, cv=cross_validation)
        grid.fit(self.x_data, self.y_data)

        # build grid search dict with results and metadata
        grid_search_ = dict(
            svm_=svm_,
            range_=range_,
            cv_results_=grid.cv_results_,
            best_score=grid.best_score_,
            best_params=grid.best_params_
        )

        # plot heatmap if defined
        if heatmap_:
            if 'norm' in heatmap_:
                self.draw_heatmap(grid_search_, norm=heatmap_['norm'])
            else:
                self.draw_heatmap(grid_search_)
            self.fig.savefig(heatmap_['filename'])
            self.fig_reset()

        return grid.best_params_, grid.best_score_

    def draw_heatmap(self, grid_search, norm=None):
        # clear previous heatmap
        self.ax.clear()

        # adjust subplot
        self.fig.subplots_adjust(left=.05, right=0.95, bottom=0.15, top=0.90)

        # svm metadata
        impl = grid_search['svm_']['impl']
        kernel = grid_search['svm_']['kernel']

        # parameter ranges
        if impl in ['svc', 'SVC', 'c_svc']:
            svm_param_name = 'C'
            svm_param_range = grid_search['range_']['C']
        elif impl in ['nusvc', 'NuSVC', 'nu_svc']:
            svm_param_name = 'nu'
            svm_param_range = grid_search['range_']['nu']
        else:
            print('svmlab: Unrecognized svm type')
            assert False
        gamma_range = grid_search['range_']['gamma']

        # cross-validation results
        cv_results_ = grid_search['cv_results_']

        scores = cv_results_['mean_test_score'].reshape(len(svm_param_range), len(gamma_range))
        vmin = scores.min()
        midpoint = (scores.max() + scores.min()) * 0.5

        # best params and score
        best_score = grid_search['best_score']
        best_params = grid_search['best_params']

        # highlight best params with mask that filters out all except best score
        mask_scores = np.ma.masked_where(scores != best_score, scores)   
        masked_scores = np.ma.masked_array(scores, mask_scores.mask)

        # heatmap
        heatmap = self.ax.imshow(scores, interpolation='nearest', cmap=plt.cm.hot, norm=norm)
        #self.ax.imshow(masked_scores, interpolation='nearest', cmap=plt.cm.bwr, alpha=0.5)
        self.ax.set_xlabel('gamma')
        self.ax.set_ylabel(svm_param_name)
        self.ax.set_xticks(np.arange(len(gamma_range)))
        self.ax.set_xticklabels(gamma_range)
        self.ax.set_yticks(np.arange(len(svm_param_range)))
        self.ax.set_yticklabels(svm_param_range)
        #self.ax.tick_params(axis='x', labelrotation=45)
        for tick in self.ax.get_xticklabels():
            tick.set_rotation(45)
        self.ax.set_title('Validation Accuracy')

        # colorbar
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(heatmap, cax=cax, orientation='vertical')

        return

    def draw_plot(self, result):
        # clear previous plot
        self.ax.clear()

        # classifier metadata 
        metadata = result['metadata']
        impl = metadata['_impl']
        kernel = metadata['kernel']
        C, nu, gamma = metadata['C'], metadata['nu'], metadata['gamma']

        # get plot data
        x_train, y_train = result['training set']
        #x_test, y_test = result['testing set']
        x_support = result['support']
        XX, YY, Z = result['contour']
        #score = result['score']
        x_min, x_max, y_min, y_max = result['bounds']

        # update plot
        # training set
        train_scat = self.ax.scatter(x_train[:,0], x_train[:,1], c=y_train,
                        zorder=10, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
        # testing set
#        test_scat = self.ax.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='v',
#                        zorder=5, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
        # supports
        supp_scat = self.ax.scatter(x_support[:,0], x_support[:,1],
                        facecolors='none', zorder=10, edgecolor='k', s=80)
        self.ax.contourf(XX, YY, Z, cmap=plt.cm.coolwarm)

        # title
        title = kernel + ' kernel:' 
        if impl == 'c_svc':
            title += (' C=%f' % C)
        elif impl == 'nu_svc':
            title += (' nu=%f' % nu)
        else:
            print('svmlab: Unrecognized svm type')
            assert(False)
        
        if kernel in KERNELS_WITH_GAMMA: 
            title += (', gamma=%f ' % gamma)
        self.ax.set_title(title)

        # legend
#        self.ax.legend([train_scat, test_scat, supp_scat],
#            ['training set', 'testing set', 'supports'],
#            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # annotations
#        font = {
#            'family': 'serif',
#            'color':  'darkred',
#            'weight': 'normal',
#            'size': 16,
#        }
#        self.ax.text(x_max-1.1, y_max-0.28, r'score: %4.2f' % score, fontsize=16,
#            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
#        self.ax.text(x_max-1.065, y_max-0.68, r'supports: %03d' % len(x_support), fontsize=12,
#            bbox={'facecolor':'white', 'alpha':0.8, 'pad':13.3})

        return

    def fig_reset(self):
        self.fig.clear()
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111)
      
        return

    def report(self):
        for results in self.results_buffer:
            yield results
			
