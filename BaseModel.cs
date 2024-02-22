using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public class BaseModel
    {
        public BaseModel(Dictionary<int, char> vectorMapping)
        {
            VECTOR_MAPPING = vectorMapping;
        }

        protected readonly Dictionary<int, char> VECTOR_MAPPING;

        // given an index (vector), what is the character?
        protected char DevectorizeKey(int idx) => VECTOR_MAPPING.FirstOrDefault(p => p.Key == idx).Value;

        // basicly, turn a character into its index (or vector) for comparison
        protected int VectorizeKey(char key) => VECTOR_MAPPING.FirstOrDefault(p => p.Value == key).Key;
    }
}
