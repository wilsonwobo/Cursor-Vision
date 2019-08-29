import wx
import wx.lib.inspection
from multiprocessing import freeze_support
from Cursor_Vision_Processing import Configuration

class HomeUI_Layout(wx.Frame):

        def __init__(self, parent, title):
                super(HomeUI_Layout, self).__init__(parent, style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER, title=title)

                # initializes the settings page and places the window in the centre of the screen
                self.HomeUI()
                self.Centre()

                # sets an icon to be displayed at the top right corner of the application window
                icon = wx.Icon()
                icon.CopyFromBitmap(wx.Bitmap("CVEye-32.ico", wx.BITMAP_TYPE_ANY))      
                self.SetIcon(icon)

        #------------------------------------------------------------------------------------------------------------------------------------
        # The Settings Directory
        #------------------------------------------------------------------------------------------------------------------------------------
        def HomeUI(self):
                # sets the window background colour to grey
                self.SetBackgroundColour('#f6f6f6')     

                # changes the font settings for the window
                font = self.GetFont();
                font.SetFamily(wx.FONTFAMILY_MODERN)
                font.SetWeight(wx.BOLD)
                self.SetFont(font);

                # initializes the window's box sizers
                hbox = wx.BoxSizer(wx.HORIZONTAL)
                vbox = wx.BoxSizer(wx.VERTICAL)

                # initializes the window's grid sizer
                sizer = wx.GridBagSizer(2, 3)

                # sets the text displayed on the window and its colour
                text1 = wx.StaticText(self, label="Settings Directory")
                text1.SetFont(wx.Font(11, wx.FONTFAMILY_MODERN, wx.NORMAL, wx.BOLD))
                text1.SetForegroundColour('#505050')
                sizer.Add(text1, pos=(1, 1), flag=wx.TOP|wx.LEFT|wx.BOTTOM,border=12)

                # sets the icon displayed next to the text
                icon = wx.StaticBitmap(self, bitmap=wx.Bitmap('Settings-48.ico'))
                sizer.Add(icon, pos=(1, 8), flag=wx.TOP|wx.ALIGN_RIGHT, border=0)
                vbox.Add(sizer, 1, wx.EXPAND | wx.ALL, 0)
                
                # displays a line under the text and icon
                line = wx.StaticLine(self)
                st = wx.StaticText(self)
                vbox.Add(st, 0.4, wx.EXPAND | wx.ALL, 0)
                vbox.Add(line, 0.1, wx.EXPAND | wx.ALL, 0)

                # empty spce for a uniform layout
                st1 = wx.StaticText(self)
                vbox.Add(st1, 0.4, wx.EXPAND | wx.ALL, 0)
                
                # initializes all the buttons displayed on the window
                button1 = wx.Button(self, label='Start Interface')
                button2 = wx.Button(self, label='Change Brightness And Contrast Values')
                button3 = wx.Button(self, label='Test')
                button4 = wx.Button(self, label='Quit')

                # sets the background colour of all the buttons displayed on the window
                button1.SetBackgroundColour('#f6f6f6')
                button2.SetBackgroundColour('#f6f6f6')
                button3.SetBackgroundColour('#f6f6f6')
                button4.SetBackgroundColour('#f6f6f6')

                # sets the text colour of all the buttons
                button1.SetForegroundColour('#505050')
                button2.SetForegroundColour('#505050')
                button3.SetForegroundColour('#505050')
                button4.SetForegroundColour('#505050')

                # binds button listeners to all of the buttons
                button1.Bind(wx.EVT_BUTTON, self.OnClicked)
                button2.Bind(wx.EVT_BUTTON, self.OnClicked)
                button3.Bind(wx.EVT_BUTTON, self.OnClicked)
                button4.Bind(wx.EVT_BUTTON, self.OnClicked)

                # vertically places the buttons on the window
                gs = wx.GridSizer(4, 1, 0, 0)
                gs.AddMany([(button1, 0, wx.EXPAND), (button2, 0, wx.EXPAND), (button3, 0, wx.EXPAND), (button4, 0, wx.EXPAND)])
                vbox.Add(gs, 4, wx.EXPAND | wx.ALL, 0)

                # displays an image on the left side of the window
                leftPan = wx.StaticBitmap(self, bitmap=wx.Bitmap('leftPanel.png'))

                # horizontally places the panel and box sizer on the window
                hbox.Add(leftPan, 4, wx.EXPAND | wx.ALL, 2)
                hbox.Add(vbox, 6, wx.EXPAND | wx.ALL, 4)

                # correctly fits the contents of the window
                self.SetSizer(hbox)
                hbox.Fit(self)

        def OnClicked(self, event): 
                btn = event.GetEventObject().GetLabel()
                if btn == 'Start Interface':
                        self.Close()    # hides the settings directory

                        Configuration.auto_setup(self)  # runs the tracking process with predefined values

                        HomeUI = HomeUI_Layout(None, title='Cursor Vision')
                        HomeUI.Show()   # shows the settings directory
                        
                elif btn == 'Change Brightness And Contrast Values':
                        self.Close()

                        Configuration.self_setup(self)  # runs the tracking process with the user's settings
                        
                        HomeUI = HomeUI_Layout(None, title='Cursor Vision')
                        HomeUI.Show()

                elif btn == 'Test':
                        self.Close()

                        Configuration.test_setup(self)  # runs the tracking process in test mode
                        
                        HomeUI = HomeUI_Layout(None, title='Cursor Vision')
                        HomeUI.Show()
                        
                elif btn == 'Quit':
                        self.Close()

def main():
        app = wx.App()
        HomeUI = HomeUI_Layout(None, title='Cursor Vision')     # sets the window title
        HomeUI.Show()   # shows the settings directory window
        app.MainLoop()

if __name__ == "__main__":
        freeze_support()        # adds support for python's multiprocessing module
        main()
