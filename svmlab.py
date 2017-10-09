import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm

''' 
fit classifier and collect useful data and metadata

'''
def run_classifier(clf, x_data, y_data):

    # determine type of svm classifier
    clf_type = clf._impl

    # split data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=0)
    
    # fit classifier according to training data
    clf.fit(x_train, y_train)
    
    # score the fit
    score = clf.score(x_test, y_test)
    
    # make predictions over a mesh grid to visualize classification regions
    x_min, x_max = x_data[:,0].min() - 0.5, x_data[:,0].max() + 0.5
    y_min, y_max = x_data[:,1].min() - 0.5, x_data[:,1].max() + 0.5
    step = .02
    XX, YY = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    
    # confusion matrix
    y_pred = clf.predict(x_test)
    cnf_mat = confusion_matrix(y_test, y_pred)
    # normalize
    cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

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
    
    return results

'''
support vector machines student class

'''
class SVMLab:
    def __init__(self, x_data, y_data):
        self.samples, self.labels = x_data, y_data
        self.study = []
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111)

    def assign_new_data(x_data, y_data):
        self.samples, self.labels = x_data, y_data

    def study_svc(self, kernel='rbf', dfs='ovr',
                  C_list=[1.], gamma_list=[1.]):
        # empty study list to put new results
        self.study = []

        # linear kernel does not use gamma, create arbitrary gamma_list to zip with C_list
        if kernel in ['linear', 'sigmoid']: 
            gamma_list = [0.0 for _ in range(len(C_list))]

        # C and gamma lists should have same length
        assert len(C_list) == len(gamma_list)

        # iterate through C, gamma tuples and develop results    
        for C, gamma in zip(C_list, gamma_list):
            if kernel in ['linear', 'sigmoid']:
                clf = svm.SVC(kernel=kernel, C=C, decision_function_shape=dfs)
            else:
                clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, decision_function_shape=dfs)

            results = run_classifier(clf, self.samples, self.labels)
            self.study.append(results)

        return self.study

    def study_nusvc(self, kernel='rbf', dfs='ovr',
                    nu_list=[1.], gamma_list=[1.]):
        # empty study list to put new results
        self.study = []

        # linear kernel does not use gamma, create arbitrary gamma_list to zip with nu_list
        if kernel in ['linear', 'sigmoid']: 
            gamma_list = [0.0 for _ in range(len(nu_list))]

        # nu and gamma lists should have same length
        assert len(nu_list) == len(gamma_list)

        # iterate through nu, gamma tuples and develop results    
        for nu, gamma in zip(nu_list, gamma_list):
            if nu > 0.9:
                print('svmlab: study cut short due to infeasible nu value (greater than 0.9)')
                break
            if kernel in ['linear', 'sigmoid']:
                clf = svm.NuSVC(kernel=kernel, nu=nu, decision_function_shape=dfs)
            else:
                clf = svm.NuSVC(kernel=kernel, nu=nu, gamma=gamma, decision_function_shape=dfs)

            results = run_classifier(clf, self.samples, self.labels)
            self.study.append(results)

        return self.study

    def animate(self, results):
        # clear plot
        self.ax.clear()

        # classifier metadata 
        metadata = results['metadata']
        clf_type = metadata['_impl']
        kernel = metadata['kernel']
        C, nu, gamma = metadata['C'], metadata['nu'], metadata['gamma']

        # get plot data
        x_train, y_train = results['training set']
        x_test, y_test = results['testing set']
        x_support = results['support']
        XX, YY, Z = results['contour']
        score = results['score']
        x_min, x_max, y_min, y_max = results['bounds']

        # update plot
        # training set
        train_scat = self.ax.scatter(x_train[:,0], x_train[:,1], c=y_train,
                        zorder=10, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
        # testing set
        test_scat = self.ax.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='v',
                        zorder=5, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
        # supports
        supp_scat = self.ax.scatter(x_support[:,0], x_support[:,1],
                        facecolors='none', zorder=10, edgecolor='k', s=80)
        self.ax.contourf(XX, YY, Z, cmap=plt.cm.coolwarm)

        # title
        title = kernel + ' kernel:' 
        if clf_type == 'c_svc':
            title += (' C=%8.3f' % C)
        elif clf_type == 'nu_svc':
            title += (' nu=%8.3f' % nu)
        else:
            print('svmlab: Unrecognized classifier type')
        
        if kernel not in ['linear', 'sigmoid']: # linear and sigmoid kernel do not include gamma
            title += (', gamma=%8.3f ' % gamma)
        self.ax.set_title(title)

        # legend
        self.ax.legend([train_scat, test_scat, supp_scat],
            ['training set', 'testing set', 'supports'],
            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # annotations
        font = {
            'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
        }
        self.ax.text(x_max-1.1, y_max-0.28, r'score: %4.2f' % score, fontsize=16,
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        self.ax.text(x_max-1.065, y_max-0.68, r'supports: %03d' % len(x_support), fontsize=12,
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':13.3})


        return
      
    def report(self):
        for results in self.study:
            yield results

    def visualize_results(self, filename, interval=0.5):
        interval = interval #in seconds
        anim = FuncAnimation(self.fig, self.animate, self.report,
                             interval=interval*1e+3, blit=False)
        anim.save(filename)

        return
			
