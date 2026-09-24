"""Execute review cells in IPython and save real outputs, without a kernel server."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('IPYTHONDIR',str(ROOT/'outputs/ipython'))
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

def main():
    os.chdir(ROOT)
    path=ROOT/'simulation/02_run_and_review.ipynb'
    nb=nbformat.read(path,as_version=4)
    shell=InteractiveShell.instance()
    count=0
    for cell in nb.cells:
        if cell.cell_type!='code':
            continue
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Review cell {count} failed: {result.error_before_exec or result.error_in_exec}')
        outputs=[]
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for displayed in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=displayed.data,metadata=displayed.metadata))
        cell.outputs=outputs
        cell.execution_count=count
    nb.metadata['execution_method']='Cells executed in order by IPython InteractiveShell; no external kernel server required.'
    nbformat.validate(nb)
    nbformat.write(nb,path)
    print(f'Executed {count} review code cells and saved their real outputs.')

if __name__=='__main__':
    main()
