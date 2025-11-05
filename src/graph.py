from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

def view_confusion_matrix(y_pred: pd.Series, y_test: pd.Series, label: str) -> None:
    # Confusion Matrix 
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Use seaborn
    sns.heatmap(conf_matrix, 
            annot=True,
            fmt='g', 
            xticklabels=[f'{label}',f'Not {label}'],
            yticklabels=[f'{label}',f'Not {label}'])
    
    # Visual configuration
    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top') 
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()

    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
    plt.show()

