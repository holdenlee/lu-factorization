import numpy as np

def zstrip(x):
  return ('%f' % x).rstrip('0').rstrip('.')

def mathwrap(x):
  return ' $ %s $ ' % zstrip(x)

def mathbfwrap(x):
  return ' $ \\mathbf{%s} $ ' % zstrip(x)

def id(x):
  return x

def compose(f,g):
  return lambda x: f(g(x))

def compose2(f,g):
  return lambda x,y: compose(f(x,y), g(x,y))

#options=None, 
def render_matrix(array, options_f = lambda i,j: id, delimiter = "p",plus_one=False):
  maybe_plus = (lambda x: x+1) if plus_one else id
  if len(array.shape)==1:
    array = np.expand_dims(array,axis=0)
    #h = array.shape[0]
    #w=1
  #else: 
  (h,w) = array.shape
  #if options is not None:
  #  return('\\begin{pmatrix}' + '\\\\'.join(['&'.join([options[i][j](zstrip(array[i,j])) for j in range(w)]) for i in range(h)]) + '\\end{pmatrix}')
  begin_env = '\\begin{%smatrix}' % delimiter
  end_env = '\\end{%smatrix}' % delimiter
  if options_f is not None:
    # for i in range(h):
    #   for j in range(w):
    #     print(options_f(i,j))
    #     print(str(array[i,j]))
    #     print(options_f(i,j)(str(array[i,j])))
    #     print(id(str(array[i,j])))
    return(begin_env + '\\\\'.join(['&'.join([options_f(i,j)(zstrip(maybe_plus(array[i,j]))) for j in range(w)]) for i in range(h)]) + end_env)
  #return('\\begin{pmatrix}' + '\\\\'.join(['&'.join([str(y) for y in x]) for x in array]) + '\\end{pmatrix}')

def frac(x,y,f1=id,f2=id):
  return "\\frac{%s}{%s}" %(f1(zstrip(x)),f2(zstrip(y)))

def render_subtract_table(a1,a2,a3,x,y):
  l = len(a1)
  s = "\\begin{center}\\begin{tabular}{%s} %s \\end{tabular}\\end{center}" % ((l+3)*"c",
                                                  "\\\\".join([
                                                             "&".join([" ","(", "&".join(map(mathbfwrap,a1)), ")"]),
                                                             "&".join(["$ -%s $" % frac(x,y,color("blue"), color("red")), "(", "&".join(map(mathwrap,a2)), ")"]),
                                                             "\\hline " + "&".join([" ", "(", "&".join(map(mathbfwrap,a3)), ")"])
                                                  ])
                                                  )
  return s

def array_f(m,n,f):
  return np.asarray([[f(i,j) for j in range(n)] for i in range(m)])

def Eij(n,i,j,val=1.0):
  E = np.eye(n)
  E[i,j]=val
  return E

def color(color_string):
  return lambda z: "{\\color{%s}%s}" % (color_string,z)

def bf(z):
  return ("\\mathbf{%s}" % z)

def boxed(z):
  return ("\\boxed{%s}" % z)

def render_print(a_old, a, L_old,L, i, k):
  print("A=")
  print(L_old)
  print("*")
  print(a_old)

  print(a_old[i,:])
  print("-%s/%s * %s"%(a_old[i,k], a_old[k,k], a[k,:]))
  print(a[i,:])

  print("R_%d <- R_%d - %s/%s * R_%d" % (i+1, i+1, a_old[i,k], a_old[k,k], k+1))
  print(L)
  print(a)
  print("")

def render_latex(a_old, a, L_old,L, i, k, m, step=1, aip=None):
  n = a.shape[0]
  saip = "" if aip is None else "\\quad\\quad "+render_matrix(aip, options_f = lambda i1,j1: color("purple") if i1>j1 and (j1,i1)<=(k,i) else id)
  s0 = "\\begin{align*} k &= %d & i &= %d \\end{align*}" % (k+1, i+1)
  if step==3:
    s1 = "$$A=%s %s %s$$" % (render_matrix(L, options_f = lambda i1,j1: color("red") if (i1,j1) ==(i,k) else id), render_matrix(a, options_f = compose2(
      lambda i1, j1: bf if i1==i else id, 
      lambda i1,j1: boxed if (i1,j1) == (k,k) else 
                    id
      )), saip)
    return ("".join([s0,s1]))
  s1 = "$$A=%s \\boxed{%s} %s$$" % (render_matrix(L_old), render_matrix(a_old, options_f = compose2(
      lambda i1, j1: bf if i1==i else id, 
      lambda i1,j1: compose(boxed,color("red")) if (i1,j1) == (k,k) else \
                    color("blue") if (i1,j1) == (i,k) else \
                    id
      )), saip)
  s3 = render_subtract_table(a_old[i], a_old[k], a[i], a_old[i,k], a_old[k,k])
  s2 = "$$R_%d \\mapsfrom R_%d - %s R_%d$$" % (i+1, i+1, frac(a_old[i,k], a_old[k,k], color("blue"), color("red")), k+1)
  fik = lambda i1,j1: color("red") if (i1,j1)==(i,k) else id
  boldf = lambda i1, j1: bf if i1==i else id
  s4 = "$$ %s \\boxed{%s} = %s $$" % (render_matrix(Eij(n,i,k,-m), options_f=fik), render_matrix(a_old, options_f=boldf), render_matrix(a, options_f=boldf)) if step == 1 else \
       "$$ \\boxed{%s} = %s %s $$" % (render_matrix(a_old, options_f=boldf), render_matrix(Eij(n,i,k,m), options_f=fik), render_matrix(a, options_f=boldf))
  return(" ".join([s0,s1,s2,s3,s4]))

def render_latex_p(a_old, a, L_old,L, p, P, i, k, m, step=1, aip=None):
  n = a.shape[0]
  saip = "" if aip is None else "\\quad\\quad "+render_matrix(aip, options_f = lambda i1,j1: id)
    #requires p inverse to color...
  s0 = "\\begin{align*} k &= %d & i &= %d & p(%d) &= %d & p(%d) &= %d\\end{align*}" % (k+1, i+1, k+1, p[k]+1, i+1, p[i]+1)
  s1 = "$$PA=%s P \\boxed{%s} %s$$" % (render_matrix(L_old), render_matrix(a_old, options_f = compose2(
      lambda i1, j1: bf if i1==p[i] else id, 
      lambda i1,j1: compose(boxed,color("red")) if (i1,j1) == (p[k],k) else \
                    color("blue") if (i1,j1) == (p[i],k) else \
                    id
      )), saip)
  s2 = "\\begin{align*} p&= %s & P&= %s \\end{align*}" % (render_matrix(p,delimiter="B",plus_one=True), 
                                                          render_matrix(P))
  if step==3:
    s1 = "$$PA=%s P %s %s$$" % (render_matrix(L, options_f = lambda i1,j1: color("red") if (i1,j1) ==(i,k) else id), render_matrix(a, options_f = compose2(
      lambda i1, j1: bf if i1==p[i] else id, 
      lambda i1,j1: boxed if (i1,j1) == (p[k],k) else 
                    id
      )), saip)
    return ("".join([s0,s1,s2]))
  #s3 = render_subtract_table(a_old[i], a_old[k], a[i], a_old[i,k], a_old[k,k])
  s3 = "$$R_{p(%d)} \\mapsfrom R_{p(%d)} - %s R_{p(%d)}$$" % (i+1, i+1, frac(a_old[p[i],k], a_old[p[k],k], color("blue"), color("red")), k+1)
  fik = lambda i1,j1: color("red") if (i1,j1)==(i,k) else id
  boldf = lambda i1, j1: bf if i1==p[i] else id
  s4 = "$$ %s P \\boxed{%s} = P %s $$" % (render_matrix(Eij(n,i,k,-m), options_f=fik), render_matrix(a_old, options_f=boldf), render_matrix(a, options_f=boldf)) if step == 1 else \
       "$$ P \\boxed{%s} = %s P %s $$" % (render_matrix(a_old, options_f=boldf), render_matrix(Eij(n,i,k,m), options_f=fik), render_matrix(a, options_f=boldf))
  return(" ".join([s0,s1,s2,s3,s4]))

def render_p(a, L, p_old, p, P_old, P, k, to_swap, aip=None):
  n = a.shape[0]
  saip = "" if aip is None else "\\quad\\quad "+render_matrix(aip, options_f = lambda i1,j1: color("purple") if i1>j1 and j1<k else id)
  s0 = "\\begin{align*} k &= {\\color{red}%d} & p({\\color{blue}%d}) &= %d\\end{align*}" % (k+1, to_swap+1, p[to_swap]+1)
  s1 = "$$PA=%s P %s %s$$" % (render_matrix(L), render_matrix(a, options_f = lambda i1, j1: boxed if (i1,j1)==(p[k],k) else id), saip)
  s_p_old = render_matrix(p_old,delimiter="B",plus_one=True)
  s_P_old = render_matrix(P_old)
  s_p = render_matrix(p,delimiter="B",plus_one=True)
  s_P = render_matrix(P)
  s_swap = render_matrix(p_to_matrix(p_swap(n,k,to_swap)), lambda i1,j1:
                         color("purple") if (i1,j1)==(k,to_swap) or (i1,j1)==(to_swap,k) else id)
  s2 = "\\begin{align*} p&= %s & P&= %s\\\\ p_{\\text{new}} &= \\mathsf{swap}_{{\\color{red}%d},{\\color{blue}%d}} \\circ %s & P_{\\text{new}} &= %s %s\\\\ &=%s &&=%s\\end{align*}" % \
       (s_p_old,s_P_old,
        k+1,to_swap+1,s_p_old,s_swap,s_P_old,
        s_p,s_P)
  return(" ".join([s0,s1,s2]))
        

def io_file(input_filepath, output_filepath, f):
  fi = open(input_filepath, "r")
  fo = open(output_filepath, "w")
  fo.write(f(fi.read()))
  fi.close()
  fo.close()

def lu_no_pivot(a, ip=False):
  n=a.shape[0]
  L= np.eye(n)
  aip = a.copy()
  pages=["\\begin{center}LU Decomposition, No pivoting\\end{center} $$A = %s$$" % render_matrix(a)]
  for k in range(n-1):
    for i in range(k+1,n):
      a_old = a.copy()
      L_old = L.copy()
      m = a[i,k]/a[k,k]
      a[i,:] -= m*a[k,:]
      aip[i,:] -= m*a[k,:]
      L[i,k] = m
      aip[i,k] = m
      pages+=[render_latex(a_old, a,L_old,L,i,k, m,step=j, aip=aip if ip else None) for j in [1,2,3]]
  pages+=["$$A=%s %s$$" % (render_matrix(L), render_matrix(a))]
  doc = '\\end{frame}\n\\begin{frame}{}'.join(pages)
  return(doc)

def amax(indices, f):
  cur_max = None
  cur_i = None
  for i in indices:
    cur = f(i)
    if cur_max is None or cur>cur_max:
      cur_max = cur
      cur_i = i
  return i

def p_swap(n,i,j):
  return np.asarray([j if k==i else \
                     i if k==j else \
                     k for k in range(n)])

def p_compose(p1,p2):
  n = len(p1)
  return np.asarray([p2[p1[i]] for i in range(n)])

def p_to_matrix(p):
  n = len(p)
  return array_f(n,n,lambda i,j: 1 if j==p[i] else 0)

def lu_pivot(a, pivot_rule ="Partial" , ip=False):
  n=a.shape[0]
  L= np.eye(n)
  aip = a.copy()
  p = np.asarray(range(n))
  P = np.eye(n)
  pages=["\\begin{center}LU Decomposition, %s pivoting\\end{center} $$A = %s$$" % (pivot_rule, render_matrix(a))]
  for k in range(n-1):
    to_swap = amax(range(k+1,n),lambda i: a[p[i],k])
    p_old = p.copy()
    P_old = P.copy()
    p = p_compose(p_swap(n,k,to_swap),p)
    P = p_to_matrix(p)
    pages+=[render_p(a, L, p_old, p, P_old, P, k, to_swap, aip=None)]
    for i in range(k+1,n):
      a_old = a.copy()
      L_old = L.copy()
      m = a[p[i],k]/a[p[k],k]
      a[p[i],:] -= m*a[p[k],:]
      aip[p[i],:] -= m*a[p[k],:]
      L[i,k] = m
      aip[p[i],k] = m
      pages+=[render_latex_p(a_old, a,L_old,L,p,P,i,k, m,step=j, aip=aip if ip else None) for j in [1,2,3]]
  s_L = render_matrix(L)
  pages+=["\\begin{align*}PA&=%s P %s\\\\&= %s %s \\\\ P&= %s\\end{align*}" % (s_L, render_matrix(a),
                                                                               s_L, render_matrix(np.matmul(P,a)),
                                                                               render_matrix(P))]
  doc = '\\end{frame}\n\\begin{frame}{}'.join(pages)
  return(doc)

if __name__ == "__main__":
    a = np.asarray([[2.0,-1,-1],[3,3,9],[3,3,5]])
    io_file("template.tex", "lu_no_pivot.tex", lambda s: s%(lu_no_pivot(a)))
    a = np.asarray([[2.0,-1,-1],[3,3,9],[3,3,5]])
    io_file("template.tex", "lu_no_pivot_in_place.tex", lambda s: s%(lu_no_pivot(a, True)))
    a = np.asarray([[1.0,2,4],[1,0,1],[-2,2,4]])
    #print(lu_pivot(a))
    io_file("template.tex", "lu_partial_pivot.tex", lambda s: s%(lu_pivot(a)))
