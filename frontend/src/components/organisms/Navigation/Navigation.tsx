import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import editImage from '../../../assets/images/edit.png';
import groundhogDayImage from '../../../assets/images/groundhog-day.png';
import jackInTheBoxImage from '../../../assets/images/jack-in-the-box.png';
import settingsImage from '../../../assets/images/settings.png';
import societyImage from '../../../assets/images/society.png';
import editVideo from '../../../assets/videos/edit.mp4';
import groundhogDayVideo from '../../../assets/videos/groundhog-day.mp4';
import jackInTheBoxVideo from '../../../assets/videos/jack-in-the-box.mp4';
import settingsVideo from '../../../assets/videos/settings.mp4';
import societyVideo from '../../../assets/videos/society.mp4';
import NavigationItem, { type NavigationItemRef } from './NavigationItem';
import './Navigation.css';

interface NavigationProps {
  isCompact: boolean;
  onNavigate: (view: 'write' | 'graph' | 'play' | 'settings' | 'help') => void;
  variant?: 'full' | 'main' | 'utility';
}

const navigationItems = [
  {
    id: 'write',
    videoSrc: editVideo,
    imageSrc: editImage,
    label: 'Write',
    view: 'write' as const,
  },
  {
    id: 'graph',
    videoSrc: societyVideo,
    imageSrc: societyImage,
    label: 'Graph',
    view: 'graph' as const,
  },
  {
    id: 'play',
    videoSrc: jackInTheBoxVideo,
    imageSrc: jackInTheBoxImage,
    label: 'Play',
    view: 'play' as const,
  },
  {
    id: 'help',
    videoSrc: groundhogDayVideo,
    imageSrc: groundhogDayImage,
    label: 'Help',
    view: 'help' as const,
  },
  {
    id: 'settings',
    videoSrc: settingsVideo,
    imageSrc: settingsImage,
    label: 'Settings',
    view: 'settings' as const,
  },
];

type AnimationPattern = 'simultaneous' | 'left-to-right' | 'right-to-left';

const Navigation = ({
  isCompact,
  onNavigate,
  variant = 'full',
}: NavigationProps) => {
  const itemRefs = useRef<(NavigationItemRef | null)[]>([]);
  const location = useLocation();

  // Filter items based on variant
  const filteredItems = navigationItems.filter((item) => {
    if (variant === 'full') return true;
    if (variant === 'main') return ['write', 'graph', 'play'].includes(item.id);
    if (variant === 'utility') return ['help', 'settings'].includes(item.id);
    return true;
  });

  const calculateDelays = (pattern: AnimationPattern): number[] => {
    const count = filteredItems.length;

    switch (pattern) {
      case 'simultaneous':
        return new Array(count).fill(0);
      case 'left-to-right':
        return filteredItems.map((_, index) => index * 200);
      case 'right-to-left':
        return filteredItems.map((_, index) => (count - 1 - index) * 200);
      default:
        return new Array(count).fill(0);
    }
  };

  const playAnimations = (pattern: AnimationPattern = 'simultaneous') => {
    const delays = calculateDelays(pattern);

    itemRefs.current.forEach((ref, index) => {
      if (ref) {
        const delay = delays[index];
        setTimeout(() => {
          ref.playVideo();
        }, delay);
      }
    });
  };

  useEffect(() => {
    if (!isCompact) {
      // Play on mount
      playAnimations('left-to-right');

      // Set up interval
      const interval = setInterval(() => {
        const patterns: AnimationPattern[] = [
          'simultaneous',
          'left-to-right',
          'right-to-left',
        ];
        const pattern = patterns[Math.floor(Math.random() * patterns.length)];
        playAnimations(pattern);
      }, 10000);

      return () => clearInterval(interval);
    }
  }, [isCompact]);

  const className = `navigation ${isCompact ? 'navigation-compact' : ''} navigation-${variant}`;

  return (
    <nav className={className}>
      {filteredItems.map((item, index) => (
        <NavigationItem
          key={item.id}
          ref={(el) => {
            itemRefs.current[index] = el;
          }}
          videoSrc={item.videoSrc}
          imageSrc={item.imageSrc}
          label={item.label}
          isCompact={isCompact}
          onClick={() => onNavigate(item.view)}
          isActive={
            location.pathname === `/${item.view}` ||
            (item.view === 'write' && location.pathname === '/')
          }
        />
      ))}
    </nav>
  );
};

export default Navigation;
