import { useNavigate } from 'react-router-dom';
import { useApp } from '../../App';
import Navigation from '../../components/organisms/Navigation/Navigation';

const HomeView = () => {
  const navigate = useNavigate();
  const { openSettings, openHelp } = useApp();

  const handleNavigate = (
    view: 'write' | 'graph' | 'play' | 'settings' | 'help'
  ) => {
    if (view === 'settings') {
      openSettings();
    } else if (view === 'help') {
      openHelp();
    } else {
      navigate(`/${view}`);
    }
  };

  return (
    <div className="home-view">
      <Navigation isCompact={false} onNavigate={handleNavigate} />
      <h1 className="animated-title home-title">worldbuilder</h1>
    </div>
  );
};

export default HomeView;
