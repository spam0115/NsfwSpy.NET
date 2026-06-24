using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NsfwSpyNS
{
    /// <summary>
    /// The result from classifying a Gif file.
    /// </summary>
    public class NsfwSpyFramesResult
    {
        /// <summary>
        /// The NsfwSpyResults for each of the frames classified with the key being the frame index.
        /// </summary>
        public IEnumerable<NsfwSpyResult> Frames { get; set; }

        /// <summary>
        /// The amount of frames classified.
        /// </summary>
        private int _FrameCount = 0;
        public int FrameCount {
            get
            {
                if (_FrameCount == 0) _FrameCount = Frames.Count();
                return _FrameCount;
            }
        }

        /// <summary>
        /// True if any of the frames have been classified as NSFW.
        /// </summary>
        public bool IsNsfw => Frames.Any(f => f.IsNsfw);

        public byte[]? Hash { get; set; } = new byte[16];
        
        public NsfwSpyFramesResult(IEnumerable<NsfwSpyResult> frames)
        {
            Frames = frames;
        }
    }
}
