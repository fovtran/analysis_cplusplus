import SchemDraw as schem
import SchemDraw.elements as e
d = schem.Drawing()
d.add( e.RES, label='100K$\Omega$' )
d.add( e.CAP, d='down', botlabel='0.1$\mu$F' )
d.draw()