import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm

''' 
run support vector classifier algorithm, return results and useful info

'''
def svc(x_data, y_data,
          kernel='rbf', C=1.0, gamma='auto', dfs='ovr'):
    # split data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=0)
    
    # initialize svm and fit
    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, decision_function_shape=dfs)
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
    
    # gamma is not relevant in linear kernel
    if kernel == 'linear':
        gamma = None 
    
    # compile results and necessary info
    results = {
        'bounds': (x_min, x_max, y_min, y_max),
        'confusion': cnf_mat, 
        'contour': (XX, YY, Z),
        'kernel': kernel,
        'parameters': (C, gamma),
        'score': score,
        'testing set': (x_test, y_test),
        'training set': (x_train, y_train)
    }
    
    return results


'''
support vector classifier student class

'''
class SVCStudent:
    def __init__(self, x_data, y_data):
        self.samples, self.labels = x_data, y_data
        self.study = []
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111)

    def study_kernel(self, kernel, range_of_experiment=range(-10, 20),
                     dynamic_C=False, dynamic_gamma=False, dfs='ovr'):
        self.study = []
        C, gamma = 1.0, 1.0
        for i in range_of_experiment:
            if dynamic_C:
                C = 10. ** (0.2 * float(i))
            if dynamic_gamma:
                gamma = 10. ** (0.2 * float(i))

            if kernel == 'linear':
                results = svc(self.samples, self.labels, kernel=kernel, C=C)
            else:
                results = svc(self.samples, self.labels, kernel=kernel, C=C, gamma=gamma)

            self.study.append(results)

        return self.study

    def animate(self, results):
        # clear plot
        self.ax.clear()

        # get latest plot data
        kernel = results['kernel']
        C, gamma = results['parameters']
        x_train, y_train = results['training set']
        x_test, y_test = results['testing set']
        XX, YY, Z = results['contour']
        score = results['score']
        x_min, x_max, y_min, y_max = results['bounds']

        # update plot
        self.ax.scatter(self.samples[:,0], self.samples[:,1], c=self.labels,
              zorder=10, cmap=plt.cm.coolwarm, edgecolor='k',s=20)
        self.ax.scatter(x_test[:,0], x_test[:,1],
              s=80, facecolors='none', zorder=10, edgecolor='k')
        self.ax.contourf(XX, YY, Z, cmap=plt.cm.coolwarm)

        # title
        title = kernel + (' kernel: C=%8.3f' % C)
        if kernel != 'linear': # only linear kernel does not include gamma
            title += (', gamma=%8.3f ' % gamma)
        self.ax.set_title(title)

        # annotations
        font = {
            'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
        }
        self.ax.text(x_max-1.1, y_max-0.28, r'score: %4.2f' % score, fontsize=16,
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

        return
      
    def publish(self):
        for results in self.study:
            yield results

    def visualize_study(self, filename, interval=0.5):
        interval = interval #in seconds
        anim = FuncAnimation(self.fig, self.animate, self.publish,
                             interval=interval*1e+3, blit=False)
        anim.save(filename)

        return
		
			

