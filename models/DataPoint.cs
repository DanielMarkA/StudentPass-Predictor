using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StudentPassPredictor.models
{
    public class DataPoint
    {
        public double[] Features { get; set; }
        public double Target { get; set; }

        public DataPoint(double[] features, double target)
        {
            Features = features;
            Target = target;
        }
    }
}
