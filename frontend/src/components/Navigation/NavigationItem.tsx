import { useRef, useImperativeHandle, forwardRef } from 'react';
import './Navigation.css';

interface NavigationItemProps {
  videoSrc: string;
  imageSrc: string;
  label: string;
  isCompact: boolean;
  onClick: () => void;
}

export interface NavigationItemRef {
  playVideo: () => void;
}

const NavigationItem = forwardRef<NavigationItemRef, NavigationItemProps>(
  ({ videoSrc, imageSrc, label, isCompact, onClick }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    useImperativeHandle(ref, () => ({
      playVideo: () => {
        if (videoRef.current && !isCompact) {
          videoRef.current.currentTime = 0;
          videoRef.current.play().catch(e => console.error('Video play error:', e));
        }
      }
    }));

    return (
      <button className="navigation-item" onClick={onClick}>
        <div className="navigation-media">
          {!isCompact ? (
            <video
              ref={videoRef}
              className="navigation-video"
              src={videoSrc}
              muted
              playsInline
            />
          ) : (
            <img
              className="navigation-image"
              src={imageSrc}
              alt={label}
            />
          )}
        </div>
        <span className="navigation-label">{label}</span>
      </button>
    );
  }
);

NavigationItem.displayName = 'NavigationItem';

export default NavigationItem;