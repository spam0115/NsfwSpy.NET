using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NsfwSpyNS
{
    /// <summary>
    /// The result from classifying an image.
    /// </summary>
    public class NsfwSpyResult
    {
        /// <summary>
        /// The drawings probability score between 0 and 1.
        /// </summary>
        public float Drawings { get; }

        /// <summary>
        /// The hentai probability score between 0 and 1.
        /// </summary>
        public float Hentai { get; }

        /// <summary>
        /// The neutral probability score between 0 and 1.
        /// </summary>
        public float Neutral { get; }

        /// <summary>
        /// The pornography probability score between 0 and 1.
        /// </summary>
        public float Pornography { get; }

        /// <summary>
        /// The sexy probability score between 0 and 1.
        /// </summary>
        public float Sexy { get; }

        /// <summary>
        /// The most likely predicted value.
        /// </summary>
        public string PredictedLabel { get; }

        /// <summary>
        /// Whether the image is likely to be explicit. True if the sum of pornography, hentai and sexy is equal to or above 0.5.
        /// </summary>
        public bool IsNsfw => Hentai + Pornography + Sexy >= 0.5;

        public NsfwSpyResult()
        {

        }

        internal NsfwSpyResult(float[] scores)
        {
            Drawings = scores[(int)EClassificationType.Drawings];
            Hentai = scores[(int)EClassificationType.Hentai];
            Neutral = scores[(int)EClassificationType.Neutral];
            Pornography = scores[(int)EClassificationType.Pornography];
            Sexy = scores[(int)EClassificationType.Sexy];

            var maxIndex = 0;
            var max = scores[0];
            for (var i = 1; i < scores.Length; i++)
            {
                if (scores[i] > max)
                {
                    max = scores[i];
                    maxIndex = i;
                }
            }
            PredictedLabel = ((EClassificationType)maxIndex).ToString();

            if (Drawings + Hentai + Neutral + Pornography + Sexy == 0)
            {
                throw new ClassificationFailedException("Classification of the file failed. Make sure the file is a valid image format (jpg, png, gif etc) and has been loaded correctly.");
            }
        }

        /// <summary>
        /// Get the classification types as a dictionary ordered by their score.
        /// </summary>
        /// <returns>Dictionary of the prediction scores.</returns>
        public Dictionary<string, float> ToDictionary()
        {
            var dictionary = new Dictionary<string, float>
            {
                { "Drawings", Drawings },
                { "Hentai", Hentai },
                { "Neutral", Neutral },
                { "Pornography", Pornography },
                { "Sexy", Sexy }
            };

            return dictionary.OrderByDescending(p => p.Value).ToDictionary(x => x.Key, x => x.Value); ;
        }
    }
}
