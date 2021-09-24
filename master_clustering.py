import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np; np.random.seed(1)
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import PySimpleGUI as sg
import sys


def create_data_frames():
    df = pd.read_html('https://www.basketball-reference.com/leagues/NBA_2021_per_game.html')[0]
    df.replace(np.nan,0,inplace=True)

    index_vals = []
    for index,words in enumerate(df['PTS']):
        if words == 'PTS':
            index_vals.append(index)

    drops = list(set(index_vals))
    df = df.drop(drops)

    for i in range(0,len(df.columns)):
        if df[df.columns[i]][0].isalpha() == False:
            try:
                df[df.columns[i]] = pd.to_numeric(df[df.columns[i]])
            except:
                pass

    df = df.sort_values(by=['PTS'],ascending=False)

    df = df.reset_index()
    df = df.drop(['index'],axis=1)

    df_names = df.copy(deep=True)

    cluster_cols = []
    for i in range(0,len(df.columns)):
        if df[df.columns[i]].dtype != 'object':
            cluster_cols.append(df.columns[i])

    layout = [
    [sg.Text('Welcome to my NBA player clustering visualization tool.')],
    [sg.Text('The first box determines what categories of player stats you want to compare')],
    [sg.Text('The next box is how many different players you want to compare.')],
    [sg.Text('Players are chosen on descending order with highest points per game first.')],
    [sg.Text('Final box is choose number of clusters')],
    [sg.Listbox(values=[c for c in cluster_cols[1:]],select_mode='multiple',key='fac',size=(30,10))],
    [sg.Text('Input number of playeres to compare. (default = 100)')],
    [sg.InputText('100',key='text')],
    [sg.Text('Number of Clusters. (default = 10 clusters)')],
    [sg.InputText('10',key='_cluster_')],
    [sg.Button('Submit'),sg.Button('All'),sg.Button('Exit')]
    ]

    window = sg.Window('Lets choose our categories',layout)

    while True:
        event,values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            sys.exit()
        if event == 'All':
            category = cluster_cols
            player_number = int(values['text'])
            cluster_num = int(values['_cluster_'])
            break
        if event == 'Submit':
            category = []
            for val in values['fac']:
                category.append(val)
            player_number = int(values['text'])
            break

    df_cluster = df[category] # look to delete

    top = df_cluster.copy(deep=True)
    top = top[0:player_number]
    Rk = []
    for i in range(0,len(top)):
        Rk.append(i)

    top['Rk'] = Rk

    df_names = df[0:player_number]

    return top,df_cluster,df_names,cluster_num

def clustering(top,cluster_num,df_names):
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(top)

    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit_transform(df_pca)
    y_kmeans = kmeans.predict(df_pca)
    centers = kmeans.cluster_centers_

    '''Add method so person can choose to see raw data for clusters'''
    cluster_df = df_names.copy(deep=True)
    cluster_df['Cluster'] = y_kmeans
    cluster_df = cluster_df.sort_values(by=['Cluster'],ascending=False)

    return df_pca,y_kmeans,cluster_df

def graphing(df_pca,k_means,df_names):
    x = df_pca[:,0]
    y = df_pca[:,1]
    names = df_names['Player']
    c=y_kmeans
    team = df_names['Tm']



    norm = plt.Normalize(1,4)
    cmap = plt.cm.Dark2

    fig,ax = plt.subplots()
    sc = plt.scatter(x,y,c=c, s=100, cmap=cmap)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join([team[n] for n in ind["ind"]]),
                               " ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

top,df_cluster,df_names,cluster_num = create_data_frames()
df_pca,y_kmeans,cluster_df = clustering(top,cluster_num,df_names)
graphing(df_pca,y_kmeans,df_names)
