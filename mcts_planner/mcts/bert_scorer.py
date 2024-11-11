from bert_score import score
import locale

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

def comparison(text, refs):
  _, _, F1 = score(text, refs, model_type="microsoft/deberta-xlarge-mnli")
  return F1

# EXAMPLE
# text = ["The robot picks up 1 plate"]
# refs =["The robot picks up 5 plates"]
# comparison(text, refs)